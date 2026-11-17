#!/usr/bin/env bash
# Prefer local .venv (Linux). Fallback: PYTHON312 / python3.12 + .deps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" "$ROOT/shape_matcher.py" "$@"
fi

PY312="${PYTHON312:-}"
if [[ -z "$PY312" ]]; then
  for c in /Users/wangzixiang/.local/bin/python3.12 "$(command -v python3.12 || true)" "$(command -v python3 || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then PY312="$c"; break; fi
  done
fi
if [[ -z "$PY312" ]]; then
  echo "Need a venv or python3. Create one: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

DEPS="$ROOT/.deps"
if [[ ! -d "$DEPS/cv2" ]]; then
  echo "Installing deps into .deps ..." >&2
  mkdir -p "$DEPS"
  "$PY312" -m pip install -r "$ROOT/requirements.txt" pyyaml --target "$DEPS"
fi

export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY312" "$ROOT/shape_matcher.py" "$@"
