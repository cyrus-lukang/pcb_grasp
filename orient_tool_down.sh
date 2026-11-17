#!/usr/bin/env bash
# 只改末端姿态（保 XYZ），不定位。
# 例:
#   ./orient_tool_down.sh --arm A --mode rx90
#   ./orient_tool_down.sh --arm A --mode rx90 --execute
#   ./orient_tool_down.sh --arm A --mode tip180 --execute
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少本仓库环境: $PY" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" "$ROOT/scripts/orient_tool_down.py" "$@"
