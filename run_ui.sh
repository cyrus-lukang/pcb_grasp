#!/usr/bin/env bash
# pcb_grasp 全流程网页控制台
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少本仓库环境: $PY" >&2
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ':8770 '; then
  echo "网页已在 http://127.0.0.1:8770 ，先停掉再启动" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" -u "$ROOT/ui/server.py"
