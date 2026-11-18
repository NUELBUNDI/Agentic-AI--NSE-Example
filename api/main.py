"""api/main.py - a small web API over the NSE research agent (FastAPI), plus a playground page.

Run it from the kit folder:
    uvicorn api.main:app --reload
Then open:
    http://127.0.0.1:8000/        the playground (a web page for demos)
    http://127.0.0.1:8000/docs    try every endpoint by hand

Endpoints:
    GET  /health                        is it up? where does data come from?
    GET  /tickers                       tickers you can ask about
    GET  /stocks/{ticker}               price data      (tool 1, no AI)
    GET  /stocks/{ticker}/news          recent news     (tool 2, no AI)
    POST /briefs                        start the agent on a brief (runs in the background)
    GET  /briefs/{thread_id}            progress: plan, steps, and the draft once it waits for you
    POST /briefs/{thread_id}/approve    approve -> the agent publishes the report
    POST /briefs/{thread_id}/reject     reject with feedback -> the agent revises it
    POST /briefs/{thread_id}/message    tell the agent something more (e.g. "publish it now")
    GET  /reports/{ticker}              the published report (Markdown)
    GET  /reports/{ticker}/pdf          the published report (PDF)

A brief goes:  running -> awaiting_approval -> (reject -> running -> awaiting_approval)* -> approve -> running -> finished

Educational research, not investment advice.
"""
from __future__ import annotations

import base64
import os
import secrets
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # api/ -> kit root, so we can import agent etc.

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import agent as A
import helpers
import nse_data as nse

PLAYGROUND = Path(__file__).with_name("playground.html")

app = FastAPI(title="NSE Research Brief API",
              description="Stock data, news and AI-written research briefs for NSE (Kenya) stocks. "
                          "Educational research, not investment advice.")

# ---------------------------------------------------------------- password (for a public server)
PASSWORD = os.getenv("PLAYGROUND_PASSWORD", "")
ON_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))      # Railway sets this on every deploy


def _given_password(request: Request) -> str:
    """The password from the browser's login box (HTTP Basic auth; the username is ignored)."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("basic "):
        return ""
    try:
        return base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8").split(":", 1)[1]
    except Exception:
        return ""


@app.middleware("http")
async def require_password(request: Request, call_next):
    """On a public server anyone could spend your Gemini and NSE quota, so:
    - PLAYGROUND_PASSWORD set   -> every page and endpoint asks for it (except /health).
    - not set, on Railway       -> refuse to serve (never run open by accident).
    - not set, on your laptop   -> open, as before."""
    if request.url.path == "/health":                     # Railway's health check must get through
        return await call_next(request)
    if not PASSWORD:
        if ON_RAILWAY:
            return PlainTextResponse("Set the PLAYGROUND_PASSWORD variable in Railway to use this app.", 503)
        return await call_next(request)
    if not secrets.compare_digest(_given_password(request).encode(), PASSWORD.encode()):
        return Response("Password required.", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="NSE Research Playground"'})
    return await call_next(request)


_agent = None          # built on first use, so the API starts even without a model key
BRIEFS: dict[str, dict] = {}     # thread_id -> {"ticker", "running", "error", "log": [progress lines]}


def get_agent():
    """Build the agent once and reuse it. The checkpointer remembers every brief by its thread_id."""
    global _agent
    if _agent is None:
        _agent = A.build_agent(checkpointer=InMemorySaver())
    return _agent


# ---------------------------------------------------------------- request bodies
class BriefRequest(BaseModel):
    ticker: str                         # e.g. "EQTY" or "Equity Bank"


class Feedback(BaseModel):
    feedback: str                       # what the agent should change


# ---------------------------------------------------------------- helpers
def _code_or_404(ticker: str) -> str:
    """Turn the ticker into an NSE code, or answer 404 if we don't know it."""
    try:
        return nse.resolve_ticker(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))


def _in_background(thread_id: str, work) -> None:
    """Run one agent step (start, approve or reject) in a background thread, so the request returns
    at once and the page can show progress. `work(log)` does the step; `log` collects progress lines."""
    brief = BRIEFS[thread_id]
    brief.update(running=True, error=None)

    def job():
        try:
            work(brief["log"].append)
        except Exception as e:                       # e.g. no GOOGLE_API_KEY, model quota exhausted
            brief["error"] = f"{type(e).__name__}: {e}"
            brief["log"].append(f"⚠️ {brief['error']}")
        finally:
            brief["running"] = False

    threading.Thread(target=job, daemon=True).start()


def _brief_status(thread_id: str) -> dict:
    """A brief's status: running / awaiting_approval (with the draft) / finished / error,
    plus the agent's plan (todos) and its progress log."""
    if thread_id not in BRIEFS:
        raise HTTPException(404, f"No brief with thread_id '{thread_id}'.")
    brief = BRIEFS[thread_id]
    out = {"thread_id": thread_id, "ticker": brief["ticker"], "log": brief["log"]}
    if brief["running"]:
        out["status"] = "running"
    elif brief["error"]:
        out.update(status="error", error=brief["error"])
    state = get_agent().get_state(helpers.new_config(thread_id))
    out["todos"] = (state.values or {}).get("todos") or []
    if out.get("status"):
        return out
    waiting = [req for intr in state.interrupts for req in helpers._requests(intr)]
    if waiting:
        out.update(status="awaiting_approval", draft_markdown=waiting[0].get("args", {}).get("brief_markdown", ""))
        return out

    # Finished: but did it really publish? (After a reject, the model sometimes stops without publishing.)
    code = brief["ticker"]
    report, pdf = nse.report_path(code), nse.pdf_path(code)
    published = report.exists() and report.stat().st_mtime >= brief["started"]     # this brief, not an old one
    out.update(status="finished", published=published,
               publish_result=_publish_result(state),         # what publish_brief said, e.g. an error
               agent_reply=helpers.final_answer(get_agent(), helpers.new_config(thread_id)),
               report_url=f"/reports/{code}" if published else None,
               pdf_url=f"/reports/{code}/pdf" if published and pdf.exists() else None)
    return out


def _publish_result(state) -> str | None:
    """The last answer from the publish_brief tool in this conversation (None if it never ran)."""
    for m in reversed((state.values or {}).get("messages", [])):
        if isinstance(m, ToolMessage) and m.name == "publish_brief":
            return str(m.content)
    return None


def _require_waiting(thread_id: str) -> dict:
    """Approve/reject only make sense while the brief is waiting for a human decision."""
    if _brief_status(thread_id)["status"] != "awaiting_approval":
        raise HTTPException(409, "This brief is not waiting for approval.")
    return helpers.new_config(thread_id)


# ---------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def playground():
    """The playground web page."""
    return FileResponse(PLAYGROUND)


@app.get("/health")
def health():
    """Check the API is running, and whether data comes from the live API or the sample."""
    return {"status": "ok", "data_mode": nse.mode(), "model": A.MODEL}


@app.get("/tickers")
def tickers():
    """Tickers you can ask about (only the sample ones in sample mode)."""
    codes = nse.sample_tickers() if nse.mode() != "api" else nse.all_tickers()
    return [{"code": c, "company": nse.KNOWN.get(c, (c,))[0]} for c in codes]


@app.get("/stocks/{ticker}")
def stock(ticker: str):
    """Price, returns, 52-week range, trend, risk and liquidity for one stock."""
    try:
        return nse.stock_data(_code_or_404(ticker))
    except ValueError as e:                       # e.g. no sample data for this ticker
        raise HTTPException(404, str(e))


@app.get("/stocks/{ticker}/news")
def news(ticker: str, limit: int = 8):
    """Recent news, each item tagged company-specific or market-wide."""
    try:
        return nse.search_news(_code_or_404(ticker), limit)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/briefs", status_code=202)
def start_brief(body: BriefRequest):
    """Start the agent on a brief. Returns a thread_id at once; poll GET /briefs/{thread_id}.
    The agent stops before publishing and waits for you to approve or reject."""
    code = _code_or_404(body.ticker)
    config = helpers.new_config()                 # a new conversation for this brief
    thread_id = config["configurable"]["thread_id"]
    BRIEFS[thread_id] = {"ticker": code, "running": False, "error": None, "log": [], "started": time.time()}
    _in_background(thread_id, lambda log: helpers.run(get_agent(), f"Brief me on {code}", config, log=log))
    return _brief_status(thread_id)


@app.get("/briefs/{thread_id}")
def get_brief(thread_id: str):
    """Progress of a brief: 'running', 'awaiting_approval' (with the draft), 'finished' or 'error'."""
    return _brief_status(thread_id)


@app.post("/briefs/{thread_id}/approve", status_code=202)
def approve_brief(thread_id: str):
    """Approve the draft: the agent publishes /reports/<TICKER>.md and the PDF."""
    config = _require_waiting(thread_id)
    _in_background(thread_id, lambda log: helpers.approve(get_agent(), config, log=log))
    return _brief_status(thread_id)


@app.post("/briefs/{thread_id}/reject", status_code=202)
def reject_brief(thread_id: str, body: Feedback):
    """Reject with feedback: the agent revises the draft and asks for approval again."""
    config = _require_waiting(thread_id)
    _in_background(thread_id, lambda log: helpers.reject(get_agent(), config, body.feedback, log=log))
    return _brief_status(thread_id)


class Message(BaseModel):
    text: str                           # e.g. "Publish the brief now."


@app.post("/briefs/{thread_id}/message", status_code=202)
def message_brief(thread_id: str, body: Message):
    """Send the agent a follow-up message in the same conversation, e.g. when it stopped without
    publishing. If it then calls publish_brief, the brief waits for your approval again."""
    status = _brief_status(thread_id)["status"]
    if status in ("running", "awaiting_approval"):
        raise HTTPException(409, f"The brief is {status.replace('_', ' ')}; send messages once it has finished.")
    config = helpers.new_config(thread_id)
    BRIEFS[thread_id]["started"] = time.time()
    _in_background(thread_id, lambda log: (log(f"🧑 You: {body.text}"), helpers.run(get_agent(), body.text, config, log=log)))
    return _brief_status(thread_id)


@app.get("/reports/{ticker}", response_class=PlainTextResponse)
def report_markdown(ticker: str):
    """The published report as Markdown text."""
    path = nse.report_path(_code_or_404(ticker))
    if not path.exists():
        raise HTTPException(404, "No published report yet. Create one with POST /briefs.")
    return path.read_text(encoding="utf-8")


@app.get("/reports/{ticker}/pdf")
def report_pdf(ticker: str, download: bool = False):
    """The published report as a PDF (shown in the browser; add ?download=true to save it)."""
    code = _code_or_404(ticker)
    path = nse.pdf_path(code)
    if not path.exists():
        raise HTTPException(404, "No PDF yet. Publish a brief first.")
    return FileResponse(path, media_type="application/pdf", filename=f"{code}_brief.pdf",
                        content_disposition_type="attachment" if download else "inline")
