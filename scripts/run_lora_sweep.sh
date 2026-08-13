#!/usr/bin/env bash
# Stage 8b: train and rank several LoRA seeds, sequentially.
#
# The sibling of run_seed_sweep.sh, pointed at the adapter driver. Unlike the full
# fine-tune, a LoRA seed moves the WEIGHTS as well as the batch order -- `lora_A` is
# Kaiming-random -- so this measures more than data ordering.
#
# Usage, from the repo root:
#     scripts/run_lora_sweep.sh 1 2
#     STEPS=15000 scripts/run_lora_sweep.sh 0 1 2
#
# Budget at the measured 0.909 s/step: ~3.8 h of training plus ~0.6 h of scoring per
# seed. Each half is skipped independently, so an interrupted sweep resumes rather
# than retraining or skipping a seed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
STEPS="${STEPS:-15000}"
MAX_HOURS="${MAX_HOURS:-8}"
# three candidates either side of the optimum, as the full fine-tune's sweep used;
# neighbouring saves differ by less than loss and AUPRC disagree with each other
STRIDE="${STRIDE:-3}"

if [ $# -eq 0 ]; then
    echo "usage: $0 <seed> [seed ...]" >&2
    exit 64
fi

for seed in "$@"; do
    dest="${ROOT}/motor_output/runs/lora-seed${seed}"

    if [ -e "${dest}" ]; then
        echo "=== lora seed ${seed}: ${dest} exists, not retraining"
    else
        echo "=== lora seed ${seed}: training ${STEPS} steps -> ${dest}"
        "${PYTHON}" "${ROOT}/scripts/train_motor_aki_lora.py" \
            --dest "${dest}" --seed "${seed}" \
            --total-steps "${STEPS}" --max-hours "${MAX_HOURS}"
    fi

    if [ -f "${dest}/selection/checkpoint_ranking.json" ]; then
        echo "=== lora seed ${seed}: already scored, skipping"
    else
        echo "=== lora seed ${seed}: scoring checkpoints on the validation fold"
        "${PYTHON}" "${ROOT}/scripts/score_checkpoints.py" \
            --run "${dest}" --stride "${STRIDE}"
    fi

    echo "=== lora seed ${seed}: done"
done

echo "=== lora sweep complete for seeds: $*"
