#!/bin/bash
#SBATCH --job-name=colosseum-pi05
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=./logs/pi05_droid/slurm-%j.out
#SBATCH --error=./logs/pi05_droid/slurm-%j.err

# ---------------------------------------------------------------------------
# NUS SoC SLURM worker for the lerobot/pi05_droid Colosseum policy.
#
# NOTE: --time is a SAFETY CAP ONLY. The Python worker exits as soon as the
# evaluation session finishes or a startup/idle timeout fires, so the GPU is
# released immediately -- it does not run for the full walltime.
# ---------------------------------------------------------------------------

ENV_NAME="venvs/pi05"
HOME_PATH="/home/n/ntasang"
PROJECT_PATH="${HOME_PATH}/projects/robocolosseum-host"

cd "${PROJECT_PATH}" || exit 1

LOG_DIR="./logs/pi05_droid"
mkdir -p "${LOG_DIR}"

# Activate the project virtualenv.
source "${HOME_PATH}/${ENV_NAME}/bin/activate"

# Hugging Face cache on persistent home storage so model downloads survive
# across jobs (SLURM_TMPDIR is wiped when the job ends).
export HF_HOME=${HOME_PATH}/cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Scratch tmp on local disk.
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER}/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

echo "Starting policy worker at $(date)"
echo "Running on host: $(hostname)"
echo "Using Hugging Face cache: $HF_HOME"
df -Th "$HF_HOME"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# Router credentials MUST come from the environment (never commit them).
# Set them before sbatch, or source a private, git-ignored env file here:
#   source /home/n/ntasang/secrets/colosseum.env
# The file should export COLOSSEUM_ROUTER_URL and COLOSSEUM_TOKEN.

# Dry run by default (no actions sent). The launcher sets ENABLE_ACTION=1 only
# when an operator has opted in; otherwise this stays a dry run. You can also
# export ENABLE_ACTION=1 manually before sbatch.
# Example: `ENABLE_ACTION=1 sbatch slurm/pi05_droid.sh` to run a real robot session.
ACTION_ARGS=()
if [[ "${ENABLE_ACTION:-0}" == "1" ]]; then
    echo "ACTION SENDING ENABLED (ENABLE_ACTION=1) -- the robot may move."
    ACTION_ARGS+=(--enable-action)
fi

python scripts/run_policy.py \
    --policy pi05_droid \
    --config configs/pi05_droid.yaml \
    --router-config configs/router.yaml \
    "${ACTION_ARGS[@]}"

echo "Policy worker ended at $(date)"
