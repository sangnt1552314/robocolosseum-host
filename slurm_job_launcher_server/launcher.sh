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

# --- refuse to start if another launcher is already up ---------------------
# The free ngrok plan allows only ONE online endpoint. A second launcher would
# collide with ERR_NGROK_334, so bail out early with a clear message.
OTHER_LAUNCHERS=$(squeue -h -u "${USER}" --name=colosseum-launcher -o '%A' 2>/dev/null \
    | grep -v -x "${SLURM_JOB_ID:-}" || true)
if [[ -n "${OTHER_LAUNCHERS}" ]]; then
    echo "ERROR: another colosseum-launcher job is already running: ${OTHER_LAUNCHERS//$'\n'/ }" >&2
    echo "Stop it first (scancel <job_id>) or wait for it to exit, then resubmit." >&2
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

# --- start ngrok, retrying past a stale endpoint (ERR_NGROK_334) -----------
# When a previous launcher died without a clean shutdown, ngrok's edge may keep
# the single free endpoint "online" for a short reconnect window. Retry with a
# delay until that old session expires, instead of failing the whole job.
NGROK_LOG="./logs/launcher/ngrok-${SLURM_JOB_ID}.log"
NGROK_MAX_ATTEMPTS="${NGROK_MAX_ATTEMPTS:-8}"
NGROK_RETRY_DELAY="${NGROK_RETRY_DELAY:-20}"

get_ngrok_url() {
    curl -s "${NGROK_API}" \
        | python -c 'import sys,json;
d=json.load(sys.stdin);
t=d.get("tunnels",[]);
print(t[0]["public_url"] if t else "")' 2>/dev/null
}

echo "Waiting for ngrok public URL..."
PUBLIC_URL=""
for attempt in $(seq 1 "${NGROK_MAX_ATTEMPTS}"); do
    echo "Starting ngrok (attempt ${attempt}/${NGROK_MAX_ATTEMPTS})..."
    "${NGROK_BIN}" http "${LAUNCHER_PORT}" --log=stdout >> "${NGROK_LOG}" 2>&1 &
    NGROK_PID=$!

    # Poll the local ngrok API for up to ~30s, but stop early if ngrok died.
    for _ in $(seq 1 15); do
        if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
            break
        fi
        PUBLIC_URL=$(get_ngrok_url)
        [[ -n "${PUBLIC_URL}" ]] && break
        sleep 2
    done

    if [[ -n "${PUBLIC_URL}" ]]; then
        break
    fi

    # This attempt failed; surface the likely cause and retry after a delay.
    if grep -q "ERR_NGROK_334" "${NGROK_LOG}" 2>/dev/null; then
        echo "ngrok endpoint still online from a previous session (ERR_NGROK_334)." >&2
        echo "Retrying in ${NGROK_RETRY_DELAY}s; the old session should expire soon." >&2
    else
        echo "ngrok did not come up on attempt ${attempt}; retrying in ${NGROK_RETRY_DELAY}s." >&2
    fi
    kill "${NGROK_PID}" 2>/dev/null
    wait "${NGROK_PID}" 2>/dev/null
    NGROK_PID=""
    [[ "${attempt}" -lt "${NGROK_MAX_ATTEMPTS}" ]] && sleep "${NGROK_RETRY_DELAY}"
done

if [[ -n "${PUBLIC_URL}" ]]; then
    echo "=========================================================="
    echo "LAUNCHER PUBLIC URL: ${PUBLIC_URL}"
    echo "=========================================================="
else
    echo "ERROR: ngrok never established a tunnel after ${NGROK_MAX_ATTEMPTS} attempts." >&2
    echo "If you see ERR_NGROK_334, stop the other online endpoint (scancel the old" >&2
    echo "launcher job or stop the tunnel in the ngrok dashboard). See ${NGROK_LOG}." >&2
    exit 1
fi

# --- monitor: if EITHER process exits, shut the whole launcher down --------
wait -n "${UVICORN_PID}" "${NGROK_PID}"
echo "A launcher process exited; shutting down (see logs above)."
# cleanup runs via the EXIT trap.
