#!/usr/bin/env bash
set -euo pipefail
set -m

REPO="/home/geoff/code/soft-actor-critic"
ACTIVATE_CMD="source .venv/bin/activate"
PY_CMD="python3"

cd "$REPO"
# shellcheck disable=SC1091
eval "$ACTIVATE_CMD"

ENV_ID="Pendulum-v1"
RUN_ROOT="runs"

mkdir -p "$RUN_ROOT/sac_${ENV_ID}_a16_c16"
mkdir -p "$RUN_ROOT/sac_${ENV_ID}_a32_c32"
mkdir -p "$RUN_ROOT/sac_${ENV_ID}_a64_c64"

$PY_CMD main.py \
  --env-id "$ENV_ID" \
  --actor-d-model 16 \
  --critic-d-model 16 \
  --eval-every 1000 \
  --eval-episodes 10 \
  |& tee "$RUN_ROOT/sac_${ENV_ID}_a16_c16/console.log" &

$PY_CMD main.py \
  --env-id "$ENV_ID" \
  --actor-d-model 32 \
  --critic-d-model 32 \
  --eval-every 1000 \
  --eval-episodes 10 \
  |& tee "$RUN_ROOT/sac_${ENV_ID}_a32_c32/console.log" &

$PY_CMD main.py \
  --env-id "$ENV_ID" \
  --actor-d-model 64 \
  --critic-d-model 64 \
  --eval-every 1000 \
  --eval-episodes 10 \
  |& tee "$RUN_ROOT/sac_${ENV_ID}_a64_c64/console.log" &

cleanup() {
  local pids
  pids=$(jobs -pr)
  if [[ -n "$pids" ]]; then
    # Kill each job's process group so pipelines (python + tee) are terminated.
    while read -r pid; do
      kill -- -"${pid}" 2>/dev/null || true
    done <<< "$pids"
  fi
}

trap cleanup INT TERM EXIT
wait
