#!/usr/bin/env bash
# Batch match all capture sets under ../captures with the new color_0000 model.
# Writes to <set>/match_partial_0000 (does NOT touch existing <set>/match).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CAPTURES="${CAPTURES:-$ROOT/../captures}"
MODEL="${MODEL:-$ROOT/models/d435_pcb.npz}"
OUT_NAME="${OUT_NAME:-match_partial_0000}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

# Same knobs as your example command
EDGE_MODE="${EDGE_MODE:-silhouette}"
SCALE_REL="${SCALE_REL:-0.3}"
SCALE_STEP="${SCALE_STEP:-0.1}"
ANGLE_COARSE="${ANGLE_COARSE:-10}"
ANGLE_FINE="${ANGLE_FINE:-2}"
SCORE_THRESH="${SCORE_THRESH:-4.0}"

resolve_labels() {
  local set_dir="$1"
  if [[ -d "$set_dir/labels_pcb/labels" ]]; then
    echo "$set_dir/labels_pcb/labels"
  elif [[ -d "$set_dir/labels_pcb" ]] && compgen -G "$set_dir/labels_pcb/*.txt" >/dev/null; then
    echo "$set_dir/labels_pcb"
  elif [[ -d "$set_dir/labels" ]] && compgen -G "$set_dir/labels/*.txt" >/dev/null; then
    echo "$set_dir/labels"
  else
    return 1
  fi
}

mapfile -t SETS < <(
  find "$CAPTURES" -mindepth 1 -maxdepth 1 -type d -name '260322274901*' | sort
)

echo "model=$MODEL"
echo "out_name=$OUT_NAME  (old match/ untouched)"
echo "captures=$CAPTURES"
echo "sets=${#SETS[@]}"
echo

ok=0
skip=0
fail=0
declare -a FAIL_LIST=()

for set_dir in "${SETS[@]}"; do
  name="$(basename "$set_dir")"
  img_dir="$set_dir/color"
  if [[ ! -d "$img_dir" ]]; then
    echo "[SKIP] $name  (no color/)"
    skip=$((skip + 1))
    continue
  fi
  if ! labels="$(resolve_labels "$set_dir")"; then
    echo "[SKIP] $name  (no labels)"
    skip=$((skip + 1))
    continue
  fi

  out_dir="$set_dir/$OUT_NAME"
  # Never write into legacy match/
  if [[ "$(basename "$out_dir")" == "match" ]]; then
    echo "[ERR] refusing to write to match/" >&2
    exit 1
  fi
  mkdir -p "$out_dir"

  n_img=$(find "$img_dir" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  n_lab=$(find "$labels" -maxdepth 1 -name '*.txt' | wc -l)
  log="$LOG_DIR/${name}_${OUT_NAME}.log"

  ee_args=()
  if [[ -f "$set_dir/ee_poses.json" ]]; then
    ee_args=(--ee-poses "$set_dir/ee_poses.json")
  fi
  depth_args=()
  if [[ -d "$set_dir/depth_aligned" ]]; then
    depth_args=(--depth-dir "$set_dir/depth_aligned")
  fi

  echo "======== $name  imgs=$n_img labels=$n_lab → $OUT_NAME ========"
  echo "  labels=$labels"
  echo "  out=$out_dir"
  echo "  log=$log"

  set +e
  "$ROOT/run.sh" match-dir \
    --model "$MODEL" \
    --images "$img_dir" \
    --labels "$labels" \
    --out "$out_dir" \
    --edge-mode "$EDGE_MODE" \
    --scale-rel "$SCALE_REL" --scale-step "$SCALE_STEP" \
    --angle-step-coarse "$ANGLE_COARSE" --angle-step-fine "$ANGLE_FINE" \
    --score-thresh "$SCORE_THRESH" \
    "${depth_args[@]}" \
    "${ee_args[@]}" \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[OK] $name"
    ok=$((ok + 1))
  else
    echo "[FAIL] $name rc=$rc"
    fail=$((fail + 1))
    FAIL_LIST+=("$name")
  fi
  echo
done

echo "DONE ok=$ok skip=$skip fail=$fail"
if (( fail > 0 )); then
  printf 'failed: %s\n' "${FAIL_LIST[@]}"
  exit 1
fi
