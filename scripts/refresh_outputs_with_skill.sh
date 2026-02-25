#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_ROOT="${PORTFOLIO_MANAGER_SKILL_ROOT:-$HOME/.claude/skills/portfolio-manager}"

if [[ ! -f "$SKILL_ROOT/SKILL.md" ]]; then
  echo "ERROR: portfolio-manager skill not found at: $SKILL_ROOT" >&2
  echo "Set PORTFOLIO_MANAGER_SKILL_ROOT to the correct path." >&2
  exit 2
fi

RAW_PATH="${PROJECT_ROOT}/raw.json"
OUT_DIR="${PROJECT_ROOT}/outputs"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ ! -f "$RAW_PATH" ]]; then
  echo "ERROR: raw.json not found at: $RAW_PATH" >&2
  exit 2
fi

PYTHONPATH="${SKILL_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m portfolio_manager.scripts.export_wealthsimpler_artifacts \
  --project-root "$PROJECT_ROOT" \
  --out-dir "$OUT_DIR" \
  "$@"
