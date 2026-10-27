#!/bin/bash
#SBATCH -J aggregate_results
#SBATCH -p murphy
#SBATCH -c 1
#SBATCH -t 0-02:00:00
#SBATCH --mem=16GB
#SBATCH -o aggregate_%j.out
#SBATCH -e aggregate_%j.err

# Must match the ENV_VARIANT / ACTION_BLOCK_C used by run_array.sh.
ENV_VARIANT="${ENV_VARIANT:-env_para_vanilla}"
ACTION_BLOCK_C="${ACTION_BLOCK_C:-1}"
ADAPR_ACTIVITY="${ADAPR_ACTIVITY:-on}"
ADAPR_STAYER_FRACTION="${ADAPR_STAYER_FRACTION:-0}"
ADAPR_MISSPEC="${ADAPR_MISSPEC:-none}"

cd "$SLURM_SUBMIT_DIR"

module load python
mamba activate budgeted

export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export ADAPR_PARAMS_DIR="$ENV_VARIANT"
export ADAPR_ACTIVITY ADAPR_STAYER_FRACTION ADAPR_MISSPEC
if [ -z "${RESULTS_ROOT:-}" ]; then
  RESULTS_ROOT="$(python3 -c "from testbed import resolve_testbed_spec; print(resolve_testbed_spec('${ENV_VARIANT}').results_root)")"
  RESULTS_ROOT="${SLURM_SUBMIT_DIR}/${RESULTS_ROOT}"
  if [ "${ACTION_BLOCK_C}" = "0" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_no_action_c"
  fi
fi
export RESULTS_ROOT
# Default: newest folder per seed 0..299 (300 experiments). Restrict
# to one array with:
#   export SLURM_ARRAY_JOB_ID=<array_job_id>
#   python aggregate.py --array-job-id <id>
# Every folder under RESULTS_ROOT (including stale duplicates):
#   python aggregate.py --all-runs

echo "Running aggregate.py on $(hostname)"
echo "ENV_VARIANT=${ENV_VARIANT}"
echo "ADAPR_ACTIVITY=${ADAPR_ACTIVITY}  ADAPR_STAYER_FRACTION=${ADAPR_STAYER_FRACTION}  ADAPR_MISSPEC=${ADAPR_MISSPEC}"
echo "ACTION_BLOCK_C=${ACTION_BLOCK_C}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset; latest batch>}"

python aggregate.py \
  --params-dir "$ENV_VARIANT" \
  --activity "$ADAPR_ACTIVITY" \
  --stayer-fraction "$ADAPR_STAYER_FRACTION" \
  --misspec "$ADAPR_MISSPEC"
