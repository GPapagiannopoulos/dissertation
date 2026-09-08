#!/usr/bin/env bash
# Score the 7 checkpoints run_all.sh's --stride 2 skipped, for the six runs whose
# degradation rates carry the argument. Writes to <run>/selection_dense so the
# existing selection/checkpoint_ranking.json (8 rungs) is never overwritten --
# score_checkpoints.py:188 replaces the file rather than merging into it.
# Resumable: a run whose dense ranking already exists is skipped.
set -euo pipefail
cd /home/george/PycharmProjects/dissertation
PY=.venv-modelling/bin/python
STEMS="step_004000 step_008000 step_012000 step_016000 step_020000 step_024000 step_028000"
for run in ng-aki-seed0 ng-aki-seed1 ng-aki-seed2 ng-all5-r8 ng-all4-r2-a32 ng-all4-r32-a32; do
    dest="motor_output/runs/${run}/selection_dense"
    if [ -f "${dest}/checkpoint_ranking.json" ]; then
        echo "== ${run}: already dense, skipping"
        continue
    fi
    echo "== ${run}: scoring 7 rungs -> ${dest}"
    $PY scripts/evaluate/scoring/score_checkpoints.py \
        --run "motor_output/runs/${run}" \
        --fold validation \
        --checkpoints $STEMS \
        --dest "$dest"
done
echo "ALL DONE"
