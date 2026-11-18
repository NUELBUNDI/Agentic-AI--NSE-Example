# NSE Stock Research Brief: a Deep Agents masterclass kit

Ask **"Brief me on EQTY"**. The agent plans, gathers data, delegates to sub-agents, drafts a one-page brief, has it reviewed, and asks **you** before publishing it to `workspace/reports/EQTY.md` and `EQTY.pdf`.

> **Educational research, not investment advice.** The Capital Markets Authority (CMA) Kenya regulates automated investment advice.

## The agent harness in one call
```python
create_deep_agent(
    model=chat_model(),                                 # Gemini, with a 90 s timeout per call
    tools=[search_news, publish_brief],                 # tools: plain Python functions
    system_prompt=LEAD_PROMPT,                          # one line
    subagents=SUBAGENTS,                                # delegation: technical-analyst (get_stock_data), risk-reviewer
    middleware=[TodoListMiddleware()],                  # planning: write_todos
    backend=FilesystemBackend(root_dir="workspace", virtual_mode=True),   # filesystem
    memory=["/AGENTS.md"],                              # memory: house rules
    skills=["/skills/"],                                # skills: workflow + brief template
    interrupt_on={"publish_brief": True},               # human approval
    checkpointer=InMemorySaver(),                       # state: pause + resume
)
```

## Two research tools
| Tool | Returns |
|---|---|
| `get_stock_data(ticker)` | last close (KES), 1M/3M/6M/1Y returns, 52-week range, SMA50/SMA200 trend, RSI 14, volatility, max drawdown, liquidity |
| `search_news(ticker, limit)` | recent news, de-duplicated, each item tagged company-specific or market-wide, with a link to the article |

`publish_brief` writes the final report as Markdown **and** PDF, behind the human-approval gate. Every news item in the brief links to its source article.

## What's where
```
nse_brief_kit/
├── agent.py        the finished agent: tools, prompts, sub-agents  →  python agent.py EQTY
├── nse_data.py     data layer: API clients, cache → API → sample fallback, stock_data(), search_news(), PDF
├── helpers.py      run / show_plan / show_pending / approve / reject / show_outputs
├── api/            FastAPI web API + playground page  →  uvicorn api.main:app --reload
│   ├── main.py             the endpoints
│   └── playground.html     the demo page at /
├── workspace/      the agent's filesystem ("/")
│   ├── AGENTS.md                         memory: house rules
│   ├── skills/research-brief/SKILL.md    skill: workflow + one-page template
│   ├── data/   tickers.csv · cache/ · sample/ (raw API JSON)
│   └── drafts/ · analysis/ · reviews/ · reports/
├── requirements.txt · .python-version · railway.json · .env.example
│
│   laptop only (in .gitignore, so not on GitHub or Railway):
├── notebooks/      01_starter.ipynb (6 one-line TODOs) · 02_solution.ipynb
├── practicals/     practice notebooks
├── scripts/        warm_cache.py · probe_apis.py · build_notebooks.py (run from the kit root)
├── docs/FACILITATOR_NOTES.md   run-of-show, timings, troubleshooting
└── tests/
```

## Credentials
Copy `.env.example` to `.env` and fill it in. Never put keys in a notebook or commit `.env`.

| Variable | For |
|---|---|
| `GOOGLE_API_KEY` | the model (`google_genai:gemini-3.5-flash`) |
| `NSE_PRICE_TOKEN` | price API (**Bearer token**) |
| `NSE_API_USER`, `NSE_API_PASSWORD` | news API (Basic auth) |
| `PLAYGROUND_PASSWORD` | public server only: the password for the playground and API |
| `NSE_USE_SAMPLE`, `NSE_CACHE_MAX_AGE_HOURS`, `NSE_PRICE_BASE_URL`, `NSE_MODEL` | optional, see `.env.example` |

No NSE credentials? Everything switches to the committed sample data: EQTY, SCOM, KCB, EABL and COOP. You can also force it with `nse.use_sample(True)` or `NSE_USE_SAMPLE=1`.

## Run locally (VS Code or Jupyter)
```bash
python -m venv .venv
.venv/Scripts/activate                       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name nse-brief --display-name "NSE Brief (.venv)"
copy .env.example .env                       # then fill it in
```
Open `notebooks/01_starter.ipynb` and pick the **NSE Brief (.venv)** kernel (top right in VS Code). Run the cells in order.

```bash
python agent.py EQTY              # terminal version: approve / reject via input()
python agent.py EQTY --sample     # offline
```

## Playground + web API (FastAPI, in api/)
```bash
uvicorn api.main:app --reload          # playground: http://127.0.0.1:8000/   · API docs: /docs
```
The playground shows the two tools, then the agent live: its plan, every step by the lead and the sub-agents, the draft, the **Approve / Reject** buttons, and the published PDF. A brief takes about 1.5 minutes with Gemini.

| Endpoint | Does |
|---|---|
| `GET /health` · `GET /tickers` | is it up (data mode, model) · tickers you can ask about |
| `GET /stocks/{ticker}` · `GET /stocks/{ticker}/news` | the two research tools, no AI |
| `POST /briefs` `{"ticker": "EQTY"}` | starts the agent in the background → returns `thread_id` |
| `GET /briefs/{thread_id}` | progress: `running` (plan + activity log) → `awaiting_approval` (+ draft) → `finished` / `error` |
| `POST /briefs/{thread_id}/approve` · `/reject` `{"feedback": "..."}` | the human-approval gate |
| `GET /reports/{ticker}` · `GET /reports/{ticker}/pdf` | the published report (add `?download=true` to save the PDF) |

Briefs are kept in memory, so restarting the server forgets unfinished ones. On your laptop there is no password; set `PLAYGROUND_PASSWORD` to require one.

## Deploy to Railway (via GitHub)
The GitHub repo holds only the app (see "What's where"); Railway builds it with its Python builder using `railway.json`.

1. Push the kit folder to a **private** GitHub repo (`git status` should show no `.env`, notebooks, tests or docs).
2. Railway → **New Project → Deploy from GitHub repo** → pick the repo.
3. **Variables:** `PLAYGROUND_PASSWORD`, `GOOGLE_API_KEY`, `NSE_PRICE_TOKEN`, `NSE_API_USER`, `NSE_API_PASSWORD` (same values as `.env`).
4. **Settings → Networking → Generate Domain**, open it, and log in with the password (any username).

Without `PLAYGROUND_PASSWORD` the app refuses to serve on Railway, so your keys are never open to the internet. Keep one replica (briefs live in memory). Railway's disk resets on each deploy, so published reports don't survive a redeploy.

## Before class
- `python scripts/probe_apis.py`: both APIs answer.
- `python scripts/warm_cache.py --bundle`: caches the class tickers and writes `nse_brief_kit_class.zip` (never contains `.env`).
- Run one brief with the real model, in the playground or `python agent.py EQTY`.

## Tests
```bash
python tests/test_agent_graph.py        # scripted fake model: planning, tools per sub-agent, memory/skill, files, reject→approve, parallel
python tests/test_offline.py            # no credentials + network blocked
python tests/test_api.py                # the FastAPI app end to end, PDF included (fake model, sample data)
python tests/validate_notebooks.py      # notebooks valid, every cell parses, 7 checkpoints
python tests/validate_notebooks.py --execute google_genai:gemini-3.5-flash   # really run the solution notebook
```
The tests put your own drafts and reports back when they finish. After editing tools or prompts in `agent.py`, run `python scripts/build_notebooks.py` (it rewrites both notebooks, replacing hand edits).
