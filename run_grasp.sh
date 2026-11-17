#!/usr/bin/env bash
# pcb_grasp 离线喂图入口：只用本仓库 .venv。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少本仓库环境: $PY" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" "$ROOT/run_grasp.py" "$@"
