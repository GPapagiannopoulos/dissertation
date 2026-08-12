#!/usr/bin/env bash
# Stage 6c: fine-tune the same configuration under several seeds, sequentially.
#
# The evaluation framework scores stability across seeds, and every number in the
# project so far comes from seed 0. The subject-level bootstrap on the validation
# fold measures sampling variance in the COHORT; it says nothing about the
# variance of TRAINING, which is a different quantity and, on a 135M-parameter
# fine-tune, not obviously smaller.
#
# Runs are strictly sequential. Each takes 5.4 GiB of an 8 GiB card, so two do not
# fit, and the checkpoint scoring that follows each run needs the GPU too.
#
# Usage, from the repo root:
#     scripts/run_seed_sweep.sh 1 2 3 4          # seeds to run
#     STEPS=15000 scripts/run_seed_sweep.sh 1 2  # override the budget
#
# Budget at 1.45 s/step: ~6.0 h of training plus ~1.2 h of checkpoint scoring per
# seed. Four seeds is about 29 hours. Nothing here is checkpointed across seeds,
# but each seed's outputs land before the next begins, so an interrupted sweep
# leaves every completed seed intact.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
STEPS="${STEPS:-15000}"
MAX_HOURS="${MAX_HOURS:-8}"
# Seed 0's full-fold ranking put the optimum at step 10,000, with 14,000 and
# 15,000 already overfitting and 3,000 clearly short. Three candidates either side
# of that is enough to locate each seed's own optimum; scoring five costs 24 more
# minutes per seed to resolve differences smaller than loss and AUPRC's
# disagreement with each other.
STRIDE="${STRIDE:-3}"

if [ $# -eq 0 ]; then
    echo "usage: $0 <seed> [seed ...]" >&2
    exit 64
fi

for seed in "$@"; do
    dest="${ROOT}/motor_output/runs/aki-seed${seed}"

    # The two halves are skipped independently. A sweep that dies between them --
    # as one did, on a checkpoint-loading fault after 7h22m of training -- must be
    # able to resume into the scoring rather than either redoing the training or
    # skipping the seed entirely.
    if [ -e "${dest}" ]; then
        echo "=== seed ${seed}: ${dest} exists, not retraining"
    else
        echo "=== seed ${seed}: training ${STEPS} steps -> ${dest}"
        "${PYTHON}" "${ROOT}/scripts/train_motor_aki.py" \
            --dest "${dest}" --seed "${seed}" \
            --total-steps "${STEPS}" --max-hours "${MAX_HOURS}"
    fi

    if [ -f "${dest}/selection/checkpoint_ranking.json" ]; then
        echo "=== seed ${seed}: already scored, skipping"
    else
        echo "=== seed ${seed}: scoring checkpoints on the validation fold"
        "${PYTHON}" "${ROOT}/scripts/score_checkpoints.py" \
            --run "${dest}" --stride "${STRIDE}"
    fi

    echo "=== seed ${seed}: done"
done

echo "=== sweep complete for seeds: $*"
