# WealthSimpler

WealthSimpler is a local Python project for parsing portfolio exports, enriching holdings with market data, and displaying portfolio analytics in Streamlit.

## Project structure

```text
WealthSimpler/
├── market_data.py                  # Legacy market-data helper (standalone)
├── raw.json                        # Local input data (private, gitignored)
├── scripts/
│   └── refresh_outputs_with_skill.sh  # Run ~/.claude/skills/portfolio-manager exporter
└── portfolio/
    ├── main.py                     # Pipeline CLI entry
    ├── dashboard.py                # Streamlit dashboard entry
    ├── requirements.txt            # Python dependencies
    └── data_pipeline/
        ├── parser.py               # Parse raw portfolio JSON
        ├── ticker_builder.py       # Build Yahoo-compatible tickers
        ├── market_fetch.py         # Market data retrieval
        ├── compute.py              # Position metric calculations
        ├── aggregate.py            # Account-level aggregations
        └── exposure.py             # Exposure/index bucket logic
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r portfolio/requirements.txt
```

3. Put your private input file at project root as `raw.json`.
4. Run CLI pipeline:

```bash
python -m portfolio.main
```

5. Run dashboard:

```bash
streamlit run portfolio/dashboard.py
```

## Use `portfolio-manager` Skill Artifacts

This project includes a helper script to refresh analysis artifacts in `outputs/`
using the local skill at `~/.claude/skills/portfolio-manager`.

1. Install project dependencies:

```bash
pip install -r portfolio/requirements.txt
```

2. Refresh artifacts (auto-skip when up-to-date):

```bash
./scripts/refresh_outputs_with_skill.sh
```

3. Force full rebuild:

```bash
./scripts/refresh_outputs_with_skill.sh --force
```

Notes:
- Override skill path with `PORTFOLIO_MANAGER_SKILL_ROOT` if needed.
- Generated files are written to `outputs/` (CSV + Parquet + `manifest.json`).

## GitHub prep notes

- `raw.json` is ignored to avoid uploading personal account data.
- `.venv`, `__pycache__`, and local OS artifacts are ignored.
- Before first push, confirm there are no credentials or personal files staged:

```bash
git status
git add .
git status
```

## Suggested first push

```bash
git add .
git commit -m "Initial project structure and GitHub-ready cleanup"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
