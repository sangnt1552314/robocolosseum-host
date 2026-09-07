#!/bin/bash
#SBATCH --job-name=colosseum-launcher
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/launcher/slurm-%j.out
#SBATCH --error=./logs/launcher/slurm-%j.err

# ---------------------------------------------------------------------------
# CPU-only RoboColosseum job launcher (control plane).
#
# Runs FastAPI/Uvicorn on localhost and exposes it publicly via ngrok. This job
# only submits/queries/cancels GPU policy jobs -- it never runs inference.
#
# --time is a 3-day SAFETY CAP (SoC `long` partition). Stop it early with
# `scancel <this_job_id>`; the trap below tears ngrok + Uvicorn down cleanly.
# Stopping the launcher does NOT cancel already-running GPU policy jobs.
# ---------------------------------------------------------------------------

set -u

ENV_NAME="py312"
HOME_PATH="/home/n/ntasang"
PROJECT_PATH="${PROJECT_ROOT:-${HOME_PATH}/projects/robocolosseum-host}"

# Local bind + ngrok. Override via the environment before sbatch if needed.
LAUNCHER_HOST="${LAUNCHER_HOST:-127.0.0.1}"
LAUNCHER_PORT="${LAUNCHER_PORT:-8000}"
NGROK_BIN="${NGROK_BIN:-${HOME_PATH}/bin/ngrok}"
NGROK_API="http://127.0.0.1:4040/api/tunnels"

cd "${PROJECT_PATH}" || { echo "Project path not found: ${PROJECT_PATH}"; exit 1; }

mkdir -p ./logs/launcher

# --- load secrets from a private, git-ignored env file (if present) --------
# Preferred: keep LAUNCHER_API_TOKEN out of your shell history by putting it in
# slurm_job_launcher_server/.env (chmod 600, never committed). An already-set
# environment value takes precedence and is left untouched.
LAUNCHER_ENV_FILE="${LAUNCHER_ENV_FILE:-slurm_job_launcher_server/.env}"
if [[ -z "${LAUNCHER_API_TOKEN:-}" && -f "${LAUNCHER_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${LAUNCHER_ENV_FILE}"
    set +a
fi

# --- validate required secrets (never print their values) ------------------
if [[ -z "${LAUNCHER_API_TOKEN:-}" ]]; then
    echo "ERROR: LAUNCHER_API_TOKEN is not set. Export it before sbatch, or put it in ${LAUNCHER_ENV_FILE}." >&2
    exit 1
fi
if [[ ! -x "${NGROK_BIN}" ]]; then
    echo "ERROR: ngrok binary not found or not executable: ${NGROK_BIN}" >&2
    exit 1
fi

# --- python environment ----------------------------------------------------
source "${HOME_PATH}/${ENV_NAME}/bin/activate"

echo "Launcher starting at $(date) on host $(hostname)"
echo "Binding Uvicorn to ${LAUNCHER_HOST}:${LAUNCHER_PORT}"

UVICORN_PID=""
NGROK_PID=""

cleanup() {
    echo "Cleaning up launcher at $(date)"
    [[ -n "${NGROK_PID}" ]] && kill "${NGROK_PID}" 2>/dev/null
    [[ -n "${UVICORN_PID}" ]] && kill "${UVICORN_PID}" 2>/dev/null
    wait 2>/dev/null
}
# Trap Slurm termination + normal exit; tear both children down.
trap cleanup TERM INT EXIT

# --- start Uvicorn (single worker; see README on why V1 is one process) ----
python -m uvicorn slurm_job_launcher_server.app:app \
    --host "${LAUNCHER_HOST}" \
    --port "${LAUNCHER_PORT}" \
    --workers 1 &
UVICORN_PID=$!

# --- start ngrok forwarding to the local Uvicorn port ----------------------
"${NGROK_BIN}" http "${LAUNCHER_PORT}" --log=stdout > ./logs/launcher/ngrok-${SLURM_JOB_ID}.log 2>&1 &
NGROK_PID=$!

# --- print the public ngrok URL (poll the local ngrok API) -----------------
echo "Waiting for ngrok public URL..."
PUBLIC_URL=""
for _ in $(seq 1 30); do
    PUBLIC_URL=$(curl -s "${NGROK_API}" \
        | python -c 'import sys,json;
d=json.load(sys.stdin);
t=d.get("tunnels",[]);
print(t[0]["public_url"] if t else "")' 2>/dev/null)
    if [[ -n "${PUBLIC_URL}" ]]; then
        break
    fi
    sleep 2
done

if [[ -n "${PUBLIC_URL}" ]]; then
    echo "=========================================================="
    echo "LAUNCHER PUBLIC URL: ${PUBLIC_URL}"
    echo "=========================================================="
else
    echo "WARNING: could not read ngrok public URL from ${NGROK_API}." >&2
    echo "Check ./logs/launcher/ngrok-${SLURM_JOB_ID}.log" >&2
fi

# --- monitor: if EITHER process exits, shut the whole launcher down --------
wait -n "${UVICORN_PID}" "${NGROK_PID}"
echo "A launcher process exited; shutting down (see logs above)."
# cleanup runs via the EXIT trap.
