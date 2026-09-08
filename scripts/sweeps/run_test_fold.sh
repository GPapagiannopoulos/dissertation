#!/usr/bin/env bash
# Score the FROZEN arms on the test fold, one job at a time.
#
#   bash scripts/sweeps/run_test_fold.sh
#
# The design was frozen on 2026-08-27 (REPORT_HANDOVER.md, section 7). This script
# scores exactly the checkpoints that freeze names and nothing else: each LoRA run
# contributes the checkpoint its OWN VALIDATION ladder selects by loss plus its
# `last`, and each monolithic run contributes its pre-collapse window. Naming the set
# means no other test-fold number is ever produced, so there is no fuller ladder on
# disk to re-select from after the results are seen.
#
# Test predictions go to <run>/selection_test/, never <run>/selection/ -- the default
# destination would overwrite every banked validation bundle the whole report rests on.
#
# Resumable: a run whose selection_test/checkpoint_ranking.json already exists is
# skipped, so an interrupted sweep continues rather than re-spending GPU hours.
#
# ONE JOB AT A TIME, deliberately. The 14 GB host froze on 2026-08-26 running two
# concurrent jobs; MemoryHigh throttles a single cgroup and does not protect against
# two. Do not run anything else while this is going.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/.venv-modelling/bin/python"
RUNS="${ROOT}/motor_output/runs"
FOLD="${FOLD:-test}"

# the monolithic pre-collapse window: validation loss turns sharply upward at step
# 22,000 in all three seeds, and step_002000 is still pre-convergence at AUPRC ~0.128
MONO_STEMS="step_006000 step_010000 step_014000 step_018000"

LORA_RUNS=$("${PY}" -c "
from thesis.modelling.ensemble.roster import LORA_RUNS
print(' '.join(LORA_RUNS))
")
MONO_RUNS="ng-aki-seed0 ng-aki-seed1 ng-aki-seed2"

score () {
  local run="$1"; shift
  local dest="${RUNS}/${run}/selection_test"
  if [ -e "${dest}/checkpoint_ranking.json" ]; then
    echo "== ${run}: already scored on ${FOLD}, skipping"
    return 0
  fi
  echo "== ${run}: scoring $* on ${FOLD}"
  "${PY}" "${ROOT}/scripts/evaluate/scoring/score_checkpoints.py" \
    --run "${RUNS}/${run}" \
    --checkpoints $* \
    --fold "${FOLD}" \
    --dest "${dest}"
}

echo "### LoRA arm -- each run's by-loss checkpoint and its last"
for run in ${LORA_RUNS}; do
  # the stem comes from the run's VALIDATION ladder, which is what the frozen rule
  # selects on; reading it here rather than hardcoding keeps one source of truth
  stem=$("${PY}" -c "
from thesis.modelling.ensemble.roster import by_loss_stem
print(by_loss_stem('${run}'))
")
  if [ "${stem}" = "last" ]; then
    score "${run}" last
  else
    score "${run}" "${stem}" last
  fi
done

echo
echo "### monolithic arm -- the pre-collapse window, ${MONO_STEMS}"
for run in ${MONO_RUNS}; do
  score "${run}" ${MONO_STEMS}
done

echo
echo "### baseline -- the 600-round booster"
BANK="${ROOT}/motor_output/comparison/newgrid/xgboost600_${FOLD}_predictions.npz"
if [ -e "${BANK}" ]; then
  echo "== already banked, skipping"
else
  "${PY}" "${ROOT}/scripts/evaluate/scoring/bank_baseline_predictions.py" \
    --booster "${RUNS}/ng-xgb-seed0-600" \
    --fold "${FOLD}" \
    --dest "${BANK}"
fi

echo
echo "### done -- every frozen arm is scored on ${FOLD}"
