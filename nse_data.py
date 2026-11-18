
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- folders & settings
KIT_DIR = Path(__file__).resolve().parent
WORKSPACE = KIT_DIR / "workspace"            # the agent's files live here
DATA = WORKSPACE / "data"
CACHE = DATA / "cache"                       # copies of API answers we already downloaded
SAMPLE = DATA / "sample"                     # offline snapshot shipped with the kit

PRICE_BASE_URL = "https://mlapi.databuildai.com"
NEWS_URL        = "https://research.databuildai.com/nse/news/{code}"

CACHE_MAX_AGE_HOURS = float(os.getenv("NSE_CACHE_MAX_AGE_HOURS") or "24")   # how long a cached copy stays "fresh"
NEWS_MAX = 20                                # the news API rejects more than 20
THIN_TURNOVER_KES = 5_000_000                # average daily turnover below this = hard to trade
PERIODS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}   # return periods, in calendar days

# Create every folder the kit needs, so nothing fails on a fresh machine.
for folder in (CACHE / "prices", CACHE / "news", SAMPLE / "prices", SAMPLE / "news",
               WORKSPACE / "drafts", WORKSPACE / "analysis", WORKSPACE / "reviews", WORKSPACE / "reports"):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- credentials & mode
def load_env(path: Path = KIT_DIR / ".env") -> None:
    """Read KEY=VALUE lines from the .env file into environment variables.
    A variable that is already set (for example in your terminal) is not overwritten."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()
USE_SAMPLE = os.getenv("NSE_USE_SAMPLE") == "1"


def use_sample(flag: bool = True) -> None:
    """The on/off switch for sample mode. True = use only the offline sample data."""
    global USE_SAMPLE
    USE_SAMPLE = flag


def _price_token() -> str | None:
    """The secret token for the price API (sent as 'Authorization: Bearer <token>')."""
    return os.getenv("NSE_PRICE_TOKEN") or os.getenv("NSE_ACCESS_TOKEN")


def _news_auth() -> tuple[str, str] | None:
    """The (username, password) pair for the news API, or None if either is missing."""
    user, password = os.getenv("NSE_API_USER"), os.getenv("NSE_API_PASSWORD")
    return (user, password) if user and password else None


def mode() -> str:
    """Tell us where data will come from: 'api', or 'sample' (switched on, or no credentials)."""
    if USE_SAMPLE:
        return "sample"
    if _price_token() and _news_auth():
        return "api"
    return "sample (credentials missing)"


# ---------------------------------------------------------------- tickers
# NSE code -> (company name, sector, other names the company goes by in the news)
KNOWN = {
    "EQTY": ("Equity Group Holdings PLC", "Banking", ["Equity Group", "Equity Bank"]),
    "KCB": ("KCB Group PLC", "Banking", ["KCB Group", "KCB Bank", "Kenya Commercial Bank"]),
    "SCOM": ("Safaricom PLC", "Telecom", ["Safaricom", "M-Pesa"]),
    "COOP": ("Co-operative Bank of Kenya", "Banking", ["Co-op Bank", "Co-operative Bank"]),
    "ABSA": ("Absa Bank Kenya PLC", "Banking", ["Absa Kenya", "Absa Bank"]),
    "NCBA": ("NCBA Group PLC", "Banking", ["NCBA"]),
    "SCBK": ("Standard Chartered Bank Kenya", "Banking", ["StanChart Kenya", "Standard Chartered Kenya"]),
    "SBIC": ("Stanbic Holdings PLC", "Banking", ["Stanbic"]),
    "DTK": ("Diamond Trust Bank Kenya", "Banking", ["Diamond Trust", "DTB"]),
    "IMH": ("I&M Group PLC", "Banking", ["I&M Group", "I&M Bank"]),
    "HFCB": ("HF Group PLC", "Banking", ["HF Group"]),
    "BKG": ("BK Group PLC", "Banking", ["BK Group", "Bank of Kigali"]),
    "EABL": ("East African Breweries PLC", "Manufacturing", ["East African Breweries", "EABL"]),
    "BAT": ("BAT Kenya PLC", "Manufacturing", ["BAT Kenya", "British American Tobacco Kenya"]),
    "BAMB": ("Bamburi Cement PLC", "Construction", ["Bamburi"]),
    "KEGN": ("KenGen PLC", "Energy", ["KenGen", "Kenya Electricity Generating"]),
    "KPLC": ("Kenya Power and Lighting Company", "Energy", ["Kenya Power", "KPLC"]),
    "TOTL": ("TotalEnergies Marketing Kenya", "Energy", ["TotalEnergies Kenya"]),
    "UMME": ("Umeme Ltd", "Energy", ["Umeme"]),
    "KQ": ("Kenya Airways PLC", "Commercial", ["Kenya Airways"]),
    "NMG": ("Nation Media Group", "Commercial", ["Nation Media"]),
    "NSE": ("Nairobi Securities Exchange PLC", "Investment Services", ["Nairobi Securities Exchange"]),
    "JUB": ("Jubilee Holdings", "Insurance", ["Jubilee Holdings", "Jubilee Insurance"]),
    "BRIT": ("Britam Holdings", "Insurance", ["Britam"]),
    "CIC": ("CIC Insurance Group", "Insurance", ["CIC Insurance", "CIC Group"]),
    "KNRE": ("Kenya Re-Insurance Corporation", "Insurance", ["Kenya Re", "Kenya Reinsurance"]),
    "CTUM": ("Centum Investment Company PLC", "Investment", ["Centum"]),
    "KUKZ": ("Kakuzi PLC", "Agricultural", ["Kakuzi"]),
    "SASN": ("Sasini PLC", "Agricultural", ["Sasini"]),
    "CRWN": ("Crown Paints Kenya", "Construction", ["Crown Paints"]),
}


def all_tickers() -> list[str]:
    
    """Every NSE code we know: the codes in workspace/data/tickers.csv plus the KNOWN list."""
    path = DATA / "tickers.csv"
    
    codes = set(pd.read_csv(path, dtype=str)["code"]) if path.exists() else set()
    return sorted(codes | set(KNOWN))


def sample_tickers() -> list[str]:
    """The tickers that have offline sample data (one JSON file per ticker)."""
    return sorted(p.stem for p in (SAMPLE / "prices").glob("*.json"))


def resolve_ticker(text: str) -> str:
    """Turn what the user typed into an NSE code: 'eqty', 'Equity Bank', 'Safaricom' -> 'EQTY'/'SCOM'.
    Raises ValueError if we can't recognise it."""
    text = (text or "").strip()
    if text.upper() in all_tickers():                      # already a code, e.g. "eqty"
        return text.upper()

    # Every company name and nickname, lower-cased, pointing at its code.
    names = {name.lower(): code for code, (company, _, aliases) in KNOWN.items()
             for name in [company, *aliases]}
    q = text.lower()
    if q in names:                                         # exact name, e.g. "Equity Bank"
        return names[q]
    if len(q) >= 3:
        for name, code in names.items():                   # part of a name, e.g. "Safaricom"
            if q in name:
                return code
    raise ValueError(f"Unknown NSE ticker or company '{text}'. Use a code such as EQTY, SCOM or KCB.")


# ---------------------------------------------------------------- cache -> API -> sample
def _read_json(path: Path):
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load(kind: str, code: str, fetch, refresh: bool = False) -> tuple[object, str]:
    """Get data of one kind ('prices' or 'news') for one ticker, plus a label saying where it came from.

    Order: sample (if in sample mode) -> fresh cache -> live API -> old cache -> sample.
    `fetch` is the function that calls the live API, e.g. fetch_prices."""
    
    cache_file   = CACHE / kind / f"{code}.json"
    sample_file  = SAMPLE / kind / f"{code}.json"

    # 1) Sample mode: never touch the network.
    if mode() != "api":
        
        if sample_file.exists():
            return _read_json(sample_file), "sample data (offline snapshot)"
        raise ValueError(f"No sample {kind} for {code}. Sample tickers: {sample_tickers()}")

    # 2) A cached copy that is still fresh.
    if cache_file.exists() and not refresh:
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            return _read_json(cache_file), f"{kind} API (cached)"

    # 3) The live API. Save what we get so next time it comes from the cache.
    try:
        data = fetch(code)
    except ValueError:                     # the ticker itself is unknown: a fallback won't help
        raise
    except Exception as error:             # network/API problem: fall back to an older copy
        for path, label in ((cache_file, "stale cache"), (sample_file, "sample data")):
            if path.exists():
                return _read_json(path), f"{label} ({kind} API failed: {type(error).__name__})"
        raise
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data, f"{kind} API (live)"


# ---------------------------------------------------------------- prices
def fetch_prices(code: str) -> list[dict]:
    
    """Call the price API: the full daily price history for one ticker (one row per trading day)."""
    
    base     = (os.getenv("NSE_PRICE_BASE_URL") or PRICE_BASE_URL).rstrip("/")
    response = requests.get(f"{base}/api/v1/nse/get_by_ticker", params={"code": code},
                            headers={"Authorization": f"Bearer {_price_token()}"}, timeout=20)
    
    response.raise_for_status()            # stop here on 401 / 500 etc.
    rows = response.json()
    if not rows:
        raise ValueError(f"The price API has no rows for {code}.")
    return rows


def raw_rows(ticker: str = "EQTY") -> list[dict]:
    
    """The API's rows exactly as they arrive - handy for looking at the raw data in the notebook."""
    rows, _source = _load("prices", resolve_ticker(ticker), fetch_prices)
    return rows


def to_daily(rows: list[dict]) -> pd.DataFrame:
    
    """Turn the API rows into a tidy table, one row per day, oldest first.
    The API leaves 'Turnover' empty, so we estimate it ourselves: volume x closing price."""
    
    raw = pd.DataFrame(rows)
    df  = pd.DataFrame({
        "close": raw["latest"],            # the API calls the closing price 'latest'
        "prev_close": raw["prev"],
        "high": raw["High"],
        "low": raw["Low"],
        "volume": raw["Volume"],
    })
    df.index           = pd.to_datetime(raw["Date"]).rename("date")
    df["turnover_kes"] = (df["volume"] * df["close"]).round(0)
    
    return df.sort_index()


def load_prices(ticker: str, refresh: bool = False) -> tuple[str, pd.DataFrame]:
    """Get (code, daily price table) for a ticker. df.attrs['source'] says where the data came from."""
    code = resolve_ticker(ticker)
    rows, source = _load("prices", code, fetch_prices, refresh)
    df = to_daily(rows)
    df.attrs["source"] = source
    return code, df


# ---------------------------------------------------------------- metrics
def pct_return(close: pd.Series, days: int) -> float | None:
    """% change in price over the last `days` calendar days. None if we don't have that much history."""
    start_date = close.index[-1] - pd.Timedelta(days=days)
    if start_date < close.index[0]:
        return None
    start_price = close.asof(start_date)   # last close on or before the start date
    return round((close.iloc[-1] / start_price - 1) * 100, 1)


def rsi(close: pd.Series, n: int = 14) -> float | None:
    """RSI (Relative Strength Index), 0-100: above 70 = 'overbought', below 30 = 'oversold'.
    It compares the average size of up-days with down-days over the last n days."""
    if len(close) <= n:
        return None
    change = close.diff()
    avg_gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    avg_loss = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if avg_loss == 0:                      # never went down -> maximum RSI
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 1)


def annual_vol(close: pd.Series) -> float | None:
    """Volatility: how much the price swings day to day, scaled to a year (in %). Higher = riskier."""
    daily_returns = np.log(close).diff().dropna()
    if len(daily_returns) < 10:
        return None
    return round(float(daily_returns.std() * np.sqrt(252) * 100), 1)   # ~252 trading days a year


def pct_vs(price: float, level: float | None) -> float | None:
    """How far `price` is above (+) or below (-) `level`, in %."""
    return round((price / level - 1) * 100, 1) if level else None


def stock_data(ticker: str) -> dict:
    
    """TOOL 1. Everything the brief needs about how the stock has traded, with units and a date."""
    code, df = load_prices(ticker)
    close = df["close"]
    last = float(close.iloc[-1])
    year = df[df.index > df.index[-1] - pd.Timedelta(days=365)]      # the last 12 months

    # 52-week range
    high, low = float(year["high"].max()), float(year["low"].min())

    # Trend: compare the price with its 50-day and 200-day averages (SMA = simple moving average)
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    if not (sma50 and sma200):
        trend = "not enough history for SMA200"
    elif last > sma50 > sma200:
        trend = "uptrend"
    elif last < sma50 < sma200:
        trend = "downtrend"
    else:
        trend = "mixed / sideways"

    # Risk: biggest fall from a peak during the year
    max_drawdown = float((year["close"] / year["close"].cummax() - 1).min() * 100)

    # Liquidity: is the stock easy to buy and sell?
    avg_turnover = round(float(year["turnover_kes"].mean()), -3)
    zero_days    = round(float((year["volume"] == 0).mean() * 100), 1)
    thin         = avg_turnover < THIN_TURNOVER_KES or zero_days > 10

    returns = {label: pct_return(close, days) for label, days in PERIODS.items()}
    notes = [f"Source: {df.attrs['source']}",
             "Turnover is estimated as volume x close (the API's Turnover field is empty)."]
    if None in returns.values():
        notes.append("A return is null where history is too short for that period.")

    return {
        "ticker": code,
        "as_of": f"{df.index[-1]:%Y-%m-%d}",
        "currency": "KES",
        "last_close": round(last, 2),
        "returns_pct": returns,
        "range_52w": {"high": round(high, 2), "low": round(low, 2),
                      "pct_below_high": round((1 - last / high) * 100, 1)},
        "trend": {"sma50": sma50 and round(sma50, 2), "sma200": sma200 and round(sma200, 2),
                  "price_vs_sma50_pct": pct_vs(last, sma50), "price_vs_sma200_pct": pct_vs(last, sma200),
                  "label": trend, "rsi_14": rsi(close)},
        "risk": {"volatility_20d_annualised_pct": annual_vol(close.tail(21)),
                 "volatility_1y_annualised_pct": annual_vol(year["close"]),
                 "max_drawdown_1y_pct": round(max_drawdown, 1)},
        "liquidity_1y": {"avg_daily_turnover_kes": avg_turnover,
                         "avg_daily_volume_shares": round(float(year["volume"].mean())),
                         "zero_volume_days_pct": zero_days,
                         "flag": f"THIN (avg turnover < KES {THIN_TURNOVER_KES:,} or >10% zero-volume days)"
                                 if thin else "adequate for NSE"},
        "notes": notes,
    }


# ---------------------------------------------------------------- news
def fetch_news(code: str) -> dict:
    """Call the news API: the latest headlines for one ticker (username/password login)."""
    response = requests.get(NEWS_URL.format(code=code), params={"limit": NEWS_MAX},
                            auth=_news_auth(), timeout=20)
    if response.status_code == 404:        # the API has no feed for this ticker
        raise ValueError(response.json().get("detail", f"No news feed for {code}"))
    response.raise_for_status()
    return response.json()


def _is_about(code: str, company: str, text: str) -> bool:
    """True if a headline is really about this company: it names the ticker or the company.
    (We match whole names like 'Equity Bank', not loose words like 'equity'.)"""
    names = list(KNOWN.get(code, ("", "", []))[2])
    short_name = re.sub(r"\b(PLC|Limited|Ltd|Holdings)\b\.?", "", company or "", flags=re.I).strip(" ,.")
    if short_name and short_name.upper() != code:
        names.append(short_name)
    if re.search(rf"\b{re.escape(code)}\b", text):           # ticker, case-sensitive: "KCB"
        return True
    return any(re.search(rf"\b{re.escape(name)}\b", text, re.I) for name in names)


def search_news(ticker: str, limit: int = 8) -> dict:
    """TOOL 2. Recent news for a ticker: duplicates removed, company-specific first, newest first."""
    code = resolve_ticker(ticker)
    payload, source = _load("news", code, fetch_news)
    company = payload.get("company", "")

    items, seen_urls = [], set()
    for article in payload.get("results", []):
        url = (article.get("url") or "").split("?")[0].rstrip("/")   # ignore ?tracking=... bits
        if not url or url in seen_urls:                               # skip duplicates
            continue
        seen_urls.add(url)
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        items.append({
            "published": (article.get("published_at") or "")[:10] or "unknown",
            "source": article.get("source") or "unknown",
            "tag": "company-specific" if _is_about(code, company, f"{title} {description}") else "market-wide",
            "title": title,
            "summary": re.split(r"(?<=[.!?])\s", description)[0][:180],   # first sentence only
            "url": article.get("url"),
        })

    # Sort newest first (unknown dates last), then move company-specific items to the top.
    items.sort(key=lambda i: i["published"] if i["published"][0].isdigit() else "", reverse=True)
    items.sort(key=lambda i: i["tag"] != "company-specific")
    items = items[: max(1, min(limit, NEWS_MAX))]

    n_specific = sum(i["tag"] == "company-specific" for i in items)
    return {
        "ticker": code, "company": payload.get("company"), "sector": payload.get("sector"),
        "source": source,
        "counts": {"company_specific": n_specific, "market_wide": len(items) - n_specific},
        "items": items,
        "note": "market-wide items are NOT about this company - never present them as if they were.",
    }


# ---------------------------------------------------------------- reports
def report_path(code: str) -> Path:
    """Where the published brief for a ticker is saved: workspace/reports/<CODE>.md"""
    return WORKSPACE / "reports" / f"{code}.md"


def pdf_path(code: str) -> Path:
    """Where the PDF copy of the published brief is saved: workspace/reports/<CODE>.pdf"""
    return report_path(code).with_suffix(".pdf")


# The PDF's built-in font only knows basic Latin letters, so swap "smart" punctuation for plain ones.
_PLAIN = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-",
          "…": "...", "→": "->", "≈": "~", "•": "-", " ": " "}


def save_pdf(code: str, markdown_text: str) -> Path:
    """Turn the Markdown brief into a PDF: Markdown -> HTML (markdown library) -> PDF (fpdf2)."""
    import markdown                       # imported here so the rest of the kit works without them
    from fpdf import FPDF

    html = markdown.markdown(markdown_text, extensions=["tables"])
    html = html.replace("<table>", '<table border="1" width="100%">')   # visible table grid
    for fancy, plain in _PLAIN.items():
        html = html.replace(fancy, plain)
    html = html.encode("latin-1", "replace").decode("latin-1")          # anything else -> "?"

    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.write_html(html)
    pdf.output(str(pdf_path(code)))
    return pdf_path(code)


def save_report(ticker: str, markdown: str) -> str:
    """Save the final brief as Markdown AND as a PDF, and return a short confirmation for the agent.
    If the PDF fails (e.g. the libraries aren't installed), the Markdown report is still saved."""
    code = resolve_ticker(ticker)
    report_path(code).write_text(markdown, encoding="utf-8")
    message = f"Published /reports/{code}.md ({len(markdown.split())} words)"
    try:
        save_pdf(code, markdown)
        return message + f" and /reports/{code}.pdf."
    except ImportError:
        return message + ". PDF skipped: run  pip install markdown fpdf2"
    except Exception as e:
        return message + f". PDF failed: {type(e).__name__}: {e}"
