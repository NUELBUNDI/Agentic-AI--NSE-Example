"""agent.py - the finished NSE Stock Research Brief agent (Deep Agents / LangGraph).

    python agent.py EQTY             # live data   (or: python agent.py "Equity Bank")
    python agent.py EQTY --sample    # offline sample data, no NSE credentials needed

Two research tools: get_stock_data (prices + metrics) and search_news (news search),
plus publish_brief, which sits behind the human-approval gate.

The harness fundamentals, one argument each:
    tools · planning (TodoListMiddleware) · filesystem (backend) · memory (AGENTS.md)
    skills (SKILL.md) · sub-agents · human approval (interrupt_on) · checkpointer
scripts/build_notebooks.py copies the code below into the notebooks, so they never drift.
"""
import json
import os
import sys

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import nse_data as nse

MODEL = os.getenv("NSE_MODEL") or "google_genai:gemini-3.5-flash"  # needs GOOGLE_API_KEY in .env
MODEL_TIMEOUT_S = 90       # one model call may take at most this long
MODEL_RETRIES = 2          # then retry at most twice (the default is 6, with ever-longer waits)


def chat_model():
    """The chat model with a time limit. Without one, a stuck or rate-limited Gemini call
    can freeze a run for many minutes; with it, the call fails fast with a clear error."""
    from langchain.chat_models import init_chat_model
    return init_chat_model(MODEL, timeout=MODEL_TIMEOUT_S, max_retries=MODEL_RETRIES)


# ---------------------------------------------------------------- 1. TOOLS
# A tool = a function with type hints + a docstring (the model reads both).
# Tools never raise: they return the error as text so the agent can adapt.
def get_stock_data(ticker: str) -> str:
    """Price data for an NSE stock: last close (KES), 1M/3M/6M/1Y returns, 52-week range,
    trend (SMA50/SMA200, RSI 14), volatility, max drawdown and liquidity."""
    try:
        return json.dumps(nse.stock_data(ticker))
    except Exception as e:
        return f"Error: {e}"


def search_news(ticker: str, limit: int = 8) -> str:
    """Recent news for an NSE stock, each item tagged company-specific or market-wide."""
    try:
        return json.dumps(nse.search_news(ticker, limit), ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


def publish_brief(ticker: str, brief_markdown: str) -> str:
    """Publish the final, reviewed brief to /reports/<TICKER>.md and a PDF copy, /reports/<TICKER>.pdf."""
    try:
        return nse.save_report(ticker, brief_markdown)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------- 2. PROMPTS
# Short on purpose: shared rules live in workspace/AGENTS.md (memory) and the
# workflow + template live in workspace/skills/research-brief/SKILL.md (skill).
LEAD_PROMPT = "You are the Research Lead for NSE (Kenya) stock briefs. Read and follow the research-brief skill."

TECH_PROMPT = """You are a technical analyst for NSE stocks.
Call get_stock_data, save the key numbers to /analysis/<TICKER>_technical.md,
then reply with 5 factual bullets (no advice, no predictions)."""

RISK_PROMPT = """You are the risk reviewer.
Read /AGENTS.md (the house rules) and the draft. Re-check the draft's numbers with your tools.
Write /reviews/<TICKER>.md with PASS or FIX for each rule, then reply with the FIX items (or ALL PASS)."""


# ---------------------------------------------------------------- 3. SUB-AGENTS
SUBAGENTS = [
    {"name": "technical-analyst",
     "description": "Price analysis for ONE NSE ticker. Returns 5 bullets.",
     "system_prompt": TECH_PROMPT,
     "tools": [get_stock_data]},
    {"name": "risk-reviewer",
     "description": "Checks a draft brief against the house rules. Give it the draft path.",
     "system_prompt": RISK_PROMPT,
     "tools": [get_stock_data, search_news]},
]


# ---------------------------------------------------------------- 4. THE AGENT
def build_agent(model=None, checkpointer=None):
    return create_deep_agent(
        model=model or chat_model(),
        tools=[search_news, publish_brief],
        system_prompt=LEAD_PROMPT,
        subagents=SUBAGENTS,                                   # delegation
        middleware=[TodoListMiddleware()],                     # planning (write_todos)
        backend=FilesystemBackend(root_dir=str(nse.WORKSPACE), virtual_mode=True),  # workspace/ = "/"
        memory=["/AGENTS.md"],                                 # house rules (lead; reviewer reads the file)
        skills=["/skills/"],                                   # the research-brief skill
        interrupt_on={"publish_brief": True},                  # human approval
        checkpointer=checkpointer,                             # pause + resume
    )


# ---------------------------------------------------------------- CLI
def main(argv):
    import helpers
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--sample" in argv:
        nse.use_sample(True)
    ticker = " ".join(a for a in argv if not a.startswith("--")) or "EQTY"
    print(f"Data: {nse.mode()} | model: {MODEL}")
    agent, config = build_agent(checkpointer=InMemorySaver()), helpers.new_config()
    helpers.run(agent, f"Brief me on {ticker}", config)
    while helpers.pending(agent, config):                    # human-in-the-loop gate
        helpers.show_pending(agent, config, full=True)
        if input("\n[a]pprove or [r]eject? ").strip().lower().startswith("r"):
            helpers.reject(agent, config, input("Feedback: ").strip() or "Please tighten it.")
        else:
            helpers.approve(agent, config)
    helpers.show_outputs(nse.resolve_ticker(ticker))


if __name__ == "__main__":
    main(sys.argv[1:])
