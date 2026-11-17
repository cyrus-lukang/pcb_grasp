#!/usr/bin/env bash
# 在 pcb_grasp 目录检查头部双目 FFS 深度（只用本仓库 .venv）。
#
# 例:
#   cd /home/ubuntu/workspace/pcb_grasp
#   ./check_head_depth.sh --left /path/cam0_left.jpg --right /path/cam1_right.jpg
#   ./check_head_depth.sh --demo
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少本仓库环境: $PY" >&2
  exit 1
fi

ARGS=("$@")
if [[ "${1:-}" == "--demo" ]]; then
  LEFT="${ROOT}/data/demo/cam0_left.jpg"
  RIGHT="${ROOT}/data/demo/cam1_right.jpg"
  if [[ ! -f "$LEFT" || ! -f "$RIGHT" ]]; then
    echo "缺少 demo 图: $LEFT / $RIGHT" >&2
    echo "请把一对 S36 原图放到 data/demo/ 或直接传 --left/--right" >&2
    exit 1
  fi
  OUT="${ROOT}/output/head_depth_check_demo"
  ARGS=(--left "$LEFT" --right "$RIGHT" --out "$OUT")
fi

echo "Python: $PY"
echo "cwd:    $ROOT"
exec "$PY" tools/check_head_depth.py "${ARGS[@]}"
