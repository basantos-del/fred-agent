# Fred — Personal Financial Research Analyst

Fred is a hand-rolled multi-agent financial analyst. It researches equities (and other
asset classes), builds bull/bear cases, fact-checks its own drafts against source data
using a different model brand than the one that wrote them, and tracks its own call
record over time.

This is a personal learning project as much as a tool: the goal is to build an AI agent
from scratch, a real RAG pipeline, and a hand-rolled eval suite — not to wrap an existing
agent framework.

**Not financial advice.** Fred is a personal research tool. Nothing it outputs is a
licensed investment recommendation.

## Architecture

Two ways to talk to Fred, sharing the same tools:

- **Simple tool-calling agents** (`agent.py` for Groq, `agent_claude.py` for Claude) — a
  single model in a tool-use loop, used for direct model-to-model comparison.
- **Structured pipeline** (`pipeline.py`) — a fixed multi-stage flow:
  `Triage → Researcher → Planner → Advisor → Approver`
  - **Advisor**: adversarial Bull/Bear debate — each side argues independently from the
    same data, gets one rebuttal round against the other's original case, then a neutral
    synthesis weighs both sides (and independently checks any comparative/inferential
    claim rather than assuming it's valid just because it went unrebutted).
  - **Approver**: a different model brand (Groq's `gpt-oss-20b`) than the Advisor
    (Claude) re-checks the draft holistically, *and* every discrete numeric claim in the
    draft is extracted and deterministically verified against the actual tool-call result
    it should trace back to — before the holistic check runs.
  - **Coach**: logs whether Fred's past calls were adopted, to build a track record over
    time.

**RAG (SEC filings):** 10-K text is fetched from EDGAR, chunked, embedded with Voyage
(`voyage-finance-2`), and stored in a local Chroma vector store. Retrieval widens the net
(15 candidates) then reranks down to the most relevant passages with Voyage's `rerank-3`
cross-encoder.

**Tools** (`tools.py`): SEC EDGAR filings, key financial ratios and peer averages,
portfolio context (Google Sheets — including look-through exposure into fund holdings,
not just directly-held tickers), and the Groq/Claude API wrappers.

**Eval** (`eval.py`): a hand-rolled golden-set suite — fixed test cases scored by an
LLM judge, plus a deterministic "claims verified N/M" metric — logged to a Google Sheet
for tracking over time.

**UI** (`app.py`): a Streamlit app with five tabs — Chat, Dashboard, Model Comparison,
Eval, and Coach.

## Setup

```bash
git clone https://github.com/basantos-del/fred-agent.git
cd fred-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:
GROQ_API_KEY=
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
FINNHUB_API_KEY=
GOOGLE_SERVICE_ACCOUNT_JSON=
PORTFOLIO_SHEET_ID=

`GOOGLE_SERVICE_ACCOUNT_JSON` is the full service-account JSON as a single-line string
(not a file path) — the Sheets client loads it with `json.loads()`. Share the portfolio
Google Sheet (and the Eval Runs sheet) with that service account's email address, or
`gspread` will fail to open them.

## Run

```bash
streamlit run app.py
```

## Status

## Status

Working: RAG retrieval + reranking, cross-model Approver check, deterministic claim
verification, adversarial Bull/Bear debate with rebuttal round, Claude API retry/backoff
on all direct call sites, Planner tool-name validation with skipped-request surfacing,
Triage routed through Groq, dynamic-scale/K€-formatted portfolio dashboard, and a fix for
`evaluate_recommendation()`'s missing-price bug plus a `pe_ratio_note` flag (with debate-prompt
guardrails) for near-zero-EPS P/E values.

Open: table-aware chunking and per-claim source citations (rejected an early design, no
replacement yet); the eval suite's claim-check logging needs a manual header column added
to the Eval Runs sheet; a further guardrail on the P/E-note handling (closing a
"needing-a-flag-is-itself-evidence" loophole) is written but not yet committed or
reverified; claim-extraction's derived/non-derived classification over-applies to some
checkable stats (e.g. peer medians), causing under- rather than over-verification;
recommendation track record/backtest, statistically honest scoring, and a cross-model
determinism eval category are proposed (see `claude/fred-improvement-proposals-2026-09-15.md`)
but not started.
