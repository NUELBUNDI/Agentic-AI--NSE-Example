---
name: research-brief
description: Step-by-step workflow and one-page template for an NSE stock research brief. Read this before writing any brief or comparison.
---

# Research brief workflow

1. **Plan** with `write_todos`.
2. **Price analysis**: if a `technical-analyst` sub-agent exists, delegate to it with the `task` tool.
   Otherwise call `get_stock_data` yourself.
3. Call `search_news`.
4. Write the draft to `/drafts/<TICKER>.md` using the template below.
5. If a `risk-reviewer` sub-agent exists, send it the draft path, then fix every FIX item it reports.
6. If you have the `publish_brief` tool, call it with the final brief. If the human rejects it,
   apply their feedback to the draft and call `publish_brief` again.

**Comparing several tickers:** start one `technical-analyst` per ticker *in the same turn*
(they run in parallel). Then write `/drafts/COMPARE.md` with a side-by-side table. Do not publish it.

## Template (one page)

```markdown
# <TICKER> - <Company> (<Sector>) - as of <date>
> Educational research, not investment advice.

## Snapshot                 (table: last close KES, 1M/3M/6M/1Y %, 52w high/low, % below high)
## Trend & momentum         (SMA50, SMA200, trend label, RSI 14)
## Volatility & liquidity   (20d and 1y volatility, max drawdown, avg turnover KES, liquidity flag)
## Recent news              (3-5 items, company-specific first, each with date, source and a link)
- **<date>** (*<source>*): <one-line summary> - [Read the article](<url from search_news>)
## Key risks
## What to watch
## Data notes               (sources, as-of dates, estimated turnover)
```
