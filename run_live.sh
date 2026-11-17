#!/usr/bin/env bash
# pcb_grasp 实时入口：只用本仓库 .venv，不借用其它项目环境。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少本仓库环境: $PY" >&2
  echo "请先在 pcb_grasp 下准备 .venv（不要用其它项目的 venv）。" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" "$ROOT/run_live.py" "$@"
