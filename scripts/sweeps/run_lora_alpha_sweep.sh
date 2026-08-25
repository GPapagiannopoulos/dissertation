#!/usr/bin/env bash
# Stage 8e: separate LoRA's CAPACITY from its UPDATE SCALE.
#
# Every run in the target sweep used alpha = 4r, which holds the adapter's output
# scale alpha/r fixed at 4 and varies capacity alone. That row reads:
#
#     r=8  alpha=32   scale 4    AUPRC 0.1744
#     r=16 alpha=64   scale 4    AUPRC 0.1742  (0.1799 at its own best step)
#     r=32 alpha=128  scale 4    AUPRC 0.1601
#
# so capacity beyond r=16 hurts at a fixed scale. What is untested is whether a
# larger r wants a SMALLER scale. These two runs hold alpha=32 instead, giving a
# second row through the same r=8 corner:
#
#     r=8  alpha=32   scale 4    <- already measured, shared between both rows
#     r=16 alpha=32   scale 2    <- this script
#     r=32 alpha=32   scale 1    <- this script
#
# Read down the alpha=32 column against the alpha=4r column: if r=32 recovers here,
# the collapse was the scale rather than the capacity, and the 4r convention is the
# thing to abandon. If it collapses again, capacity is genuinely the limit.
#
# Targets are fixed at the four projections that won the target sweep, so the only
# thing moving is (r, alpha).
#
# Usage, from the repo root:
#     scripts/sweeps/run_lora_alpha_sweep.sh
#     scripts/sweeps/run_lora_alpha_sweep.sh all4-r32-a32

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
SEED="${SEED:-0}"
STEPS="${STEPS:-30000}"
MAX_HOURS="${MAX_HOURS:-5.5}"
STRIDE="${STRIDE:-3}"
PREFIX="${PREFIX:-lora-cfg-}"
TARGETS="${TARGETS:-q_proj k_proj v_proj ff_proj}"

# name | r | alpha
CONFIGS=(
    "all4-r16-a32|16|32"
    "all4-r32-a32|32|32"
)

selected=("$@")

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name rank alpha <<<"${config}"

    if [ ${#selected[@]} -gt 0 ]; then
        wanted=0
        for candidate in "${selected[@]}"; do
            [ "${candidate}" = "${name}" ] && wanted=1
        done
        [ "${wanted}" -eq 0 ] && continue
    fi

    dest="${ROOT}/motor_output/runs/${PREFIX}${name}"

    if [ -e "${dest}" ]; then
        echo "=== ${name}: ${dest} exists, not retraining"
    else
        echo "=== ${name}: r=${rank} alpha=${alpha} (scale $((alpha / rank))) on ${TARGETS}, seed ${SEED}"
        # shellcheck disable=SC2086
        "${PYTHON}" "${ROOT}/scripts/train/train_motor_aki_lora.py" \
            --dest "${dest}" --seed "${SEED}" \
            --total-steps "${STEPS}" --max-hours "${MAX_HOURS}" \
            --lora-r "${rank}" --lora-alpha "${alpha}" \
            --lora-targets ${TARGETS}
    fi

    if [ -f "${dest}/selection/checkpoint_ranking.json" ]; then
        echo "=== ${name}: already scored, skipping"
    else
        echo "=== ${name}: scoring checkpoints on the validation fold"
        "${PYTHON}" "${ROOT}/scripts/evaluate/score_checkpoints.py" \
            --run "${dest}" --stride "${STRIDE}"
    fi

    echo "=== ${name}: done"
done

echo "=== lora alpha sweep complete"
