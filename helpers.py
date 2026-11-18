"""helpers.py - display + human-in-the-loop helpers for the NSE brief agent (imported, not typed).

    config = new_config()                 # a fresh conversation thread
    run(agent, "Brief me on EQTY", config)   # stream progress
    show_plan(agent, config)              # the lead's todo list
    show_pending(agent, config)           # what is waiting for approval
    reject(agent, config, "Shorter news section")   # agent revises and asks again
    approve(agent, config)                # publish
    show_outputs("EQTY")                  # files the agents wrote
"""
import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

import nse_data as nse

ICONS = {"write_todos": "📝", "task": "🤝", "publish_brief": "📤", "write_file": "💾",
         "edit_file": "✏️", "read_file": "📖", "ls": "📂"}


def new_config(thread_id: str | None = None) -> dict:
    """A config for one conversation. Same thread_id = same memory (needed to resume)."""
    return {"configurable": {"thread_id": thread_id or uuid.uuid4().hex[:8]}, "recursion_limit": 200}


def _display_md(text: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            return display(Markdown(text))
    except ImportError:
        pass
    print(text)


def _short(x, n=110) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_message(msg, who: str, out=print) -> None:
    pad = "   " if who != "lead" else ""
    tag = f"[{who}]"
    if isinstance(msg, AIMessage):
        for tc in msg.tool_calls:
            args = tc["args"]
            if tc["name"] == "write_todos":
                out(f"{pad}{tag} 📝 plan:")
                for t in args.get("todos", []):
                    out(f"{pad}      {'✅' if t.get('status') == 'completed' else '▶️' if t.get('status') == 'in_progress' else '⬜'} {t.get('content')}")
            elif tc["name"] == "task":
                out(f"{pad}{tag} 🤝 delegate → {args.get('subagent_type')}: {_short(args.get('description', ''), 90)}")
            elif tc["name"] in ("write_file", "edit_file"):
                out(f"{pad}{tag} {ICONS[tc['name']]} {tc['name']} {args.get('file_path')}")
            elif tc["name"] == "publish_brief":
                out(f"{pad}{tag} 📤 publish_brief({args.get('ticker')})  [human-approval gate]")
            else:
                out(f"{pad}{tag} {ICONS.get(tc['name'], '🔧')} {tc['name']}({_short(args, 70)})")
        if msg.text and not msg.tool_calls:
            out(f"{pad}{tag} 💬 {_short(msg.text, 400 if who == 'lead' else 220)}")
    elif isinstance(msg, ToolMessage) and msg.name not in ("write_todos",):
        out(f"{pad}{tag}    ↳ {msg.name}: {_short(str(msg.content), 120)}")


def run(agent, message, config: dict, quiet: bool = False, log=print):
    """Send a user message (str) or a resume Command and stream progress until done or paused.
    `log` receives each progress line (default: print it; the web API collects them instead)."""
    inputs = message if isinstance(message, Command) else {"messages": [{"role": "user", "content": message}]}
    names: dict[str, str] = {}      # sub-agent namespace -> sub-agent name (from AIMessage.name)
    for namespace, update in agent.stream(inputs, config, stream_mode="updates", subgraphs=True):
        for node, data in (update or {}).items():
            if node == "__interrupt__":
                if not quiet:
                    log("⏸️  Paused: waiting for human approval (see show_pending / approve / reject)")
                continue
            msgs = (data or {}).get("messages", []) if isinstance(data, dict) else []
            msgs = msgs.value if hasattr(msgs, "value") else msgs     # Overwrite wrapper
            for m in msgs if isinstance(msgs, list) else []:
                key = namespace[0] if namespace else ""
                if key and isinstance(m, AIMessage) and m.name:
                    names[key] = m.name                                # e.g. 'technical-analyst'
                if not quiet:
                    _print_message(m, names.get(key, "sub-agent") if key else "lead", log)
    return final_answer(agent, config)


def final_answer(agent, config: dict) -> str:
    msgs = agent.get_state(config).values.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.text:
            return m.text
    return ""


def show_plan(agent, config: dict) -> list:
    """Print the lead's todo list (from write_todos)."""
    todos = agent.get_state(config).values.get("todos", []) or []
    marks = {"completed": "✅", "in_progress": "▶️", "pending": "⬜"}
    for t in todos:
        print(f"{marks.get(t.get('status'), '•')} {t.get('content')}")
    if not todos:
        print("(no plan yet)")
    return todos


def pending(agent, config: dict) -> list:
    """Interrupts waiting for a human decision ([] if none)."""
    return list(agent.get_state(config).interrupts)


def _requests(intr) -> list:
    v = intr.value if isinstance(intr.value, dict) else {}
    return v.get("action_requests", [])


def show_pending(agent, config: dict, full: bool = False) -> None:
    """Show each tool call waiting for approval; render the brief markdown."""
    items = pending(agent, config)
    if not items:
        print("Nothing is waiting for approval.")
    for intr in items:
        for req in _requests(intr):
            args = req.get("args", {})
            print(f"⏸️  {req.get('name')} wants to run for ticker={args.get('ticker')!r}")
            md = args.get("brief_markdown", "")
            if md:
                _display_md(md if full or len(md) < 6000 else md[:6000] + "\n\n…(truncated)")


def _decide(agent, config: dict, decision: dict, quiet: bool = False, log=print) -> str:
    items = pending(agent, config)
    if not items:
        print("Nothing is waiting for approval.")
        return ""
    resume = {i.id: {"decisions": [decision] * max(1, len(_requests(i)))} for i in items}
    return run(agent, Command(resume=resume), config, quiet=quiet, log=log)


def approve(agent, config: dict, quiet: bool = False, log=print) -> str:
    """Approve the pending publish_brief call(s) and let the agent finish."""
    log("✅ Approved by human")
    return _decide(agent, config, {"type": "approve"}, quiet, log)


def reject(agent, config: dict, message: str, quiet: bool = False, log=print) -> str:
    """Reject with feedback: the agent revises the brief and asks for approval again."""
    log(f"❌ Rejected by human: {message}")
    return _decide(agent, config, {"type": "reject", "message": message}, quiet, log)


def show_outputs(ticker: str | None = None, show_report: bool = True) -> dict:
    """List the files the agents wrote (drafts, analysis, reviews, reports); render the report."""
    found = {}
    for folder in ("drafts", "analysis", "reviews", "reports"):
        files = sorted(f for ext in ("md", "pdf") for f in (nse.WORKSPACE / folder).glob(f"{ticker or ''}*.{ext}"))
        found[folder] = [f.name for f in files]
        print(f"/{folder}/: {', '.join(found[folder]) or '-'}")
    report = nse.report_path(ticker) if ticker else None
    if show_report and report and report.exists():
        print("\n" + "=" * 70)
        _display_md(report.read_text(encoding="utf-8"))
    return found
