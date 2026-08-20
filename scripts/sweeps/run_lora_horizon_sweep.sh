#!/usr/bin/env bash
# Stage 8f: does training AT a horizon beat transferring TO it?
#
# The 48h models transfer to 72h without retraining, because the landmark grid does
# not depend on the horizon -- only `boolean_value` does. Scored that way the whole
# ordering survives and the ensemble still beats every member:
#
#     zero-shot 72h   best single 0.1998   ensemble of 10 0.2076
#
# These runs train on the 72h labels instead, so the difference between them and the
# zero-shot numbers is what retraining is worth. Three configs rather than nine: of
# the 120 possible trios, the top eight span 0.0007 AUPRC and three members already
# capture 87% of the ten-member ensemble gain, so more would restate the point.
#
# The three chosen span three different PLACEMENTS rather than three tunings of one:
# the four projections, all five, and the output projection alone. o_proj shares no
# adapted weight with the other two, which is where a horizon change would show up
# first if it changes what the model needs to attend to.
#
# Usage, from the repo root:
#     scripts/sweeps/run_lora_horizon_sweep.sh
#     SEQUENCES=meds_output/sequences_7d PREFIX=lora-7d- scripts/sweeps/run_lora_horizon_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-modelling/bin/python"
SEED="${SEED:-0}"
STEPS="${STEPS:-30000}"
MAX_HOURS="${MAX_HOURS:-5.5}"
STRIDE="${STRIDE:-3}"
PREFIX="${PREFIX:-lora-72h-}"
# the horizon lives here: same sequences, different labels
SEQUENCES="${SEQUENCES:-meds_output/sequences_72h}"

# name | target modules | r | alpha
CONFIGS=(
    "all4-r16-a32|q_proj k_proj v_proj ff_proj|16|32"
    "all5-r8|q_proj k_proj v_proj ff_proj o_proj|8|32"
    "o-r8|o_proj|8|32"
)

selected=("$@")

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name targets rank alpha <<<"${config}"

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
        echo "=== ${name}: r=${rank} alpha=${alpha} on ${targets}, seed ${SEED}, ${SEQUENCES}"
        # shellcheck disable=SC2086
        "${PYTHON}" "${ROOT}/scripts/train/train_motor_aki_lora.py" \
            --dest "${dest}" --seed "${SEED}" \
            --sequences "${ROOT}/${SEQUENCES}" \
            --total-steps "${STEPS}" --max-hours "${MAX_HOURS}" \
            --lora-r "${rank}" --lora-alpha "${alpha}" \
            --lora-targets ${targets}
    fi

    if [ -f "${dest}/selection/checkpoint_ranking.json" ]; then
        echo "=== ${name}: already scored, skipping"
    else
        # scored against the SAME horizon it trained on, or the comparison is
        # between a matched model and a transferred metric
        echo "=== ${name}: scoring on the validation fold at this horizon"
        "${PYTHON}" "${ROOT}/scripts/evaluate/score_checkpoints.py" \
            --run "${dest}" --stride "${STRIDE}" \
            --sequences "${ROOT}/${SEQUENCES}"
    fi

    echo "=== ${name}: done"
done

echo "=== lora horizon sweep complete"
