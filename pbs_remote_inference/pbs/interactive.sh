#!/bin/bash
# Serve models from an INTERACTIVE GPU job, with no queue wait per test. Same layout as
# cohost.pbs: one colosseum_server.py per family on 127.0.0.1:9101, 9102, ... (MODELS order),
# each in its family's container + venv (serve.sh), plus an optional tunnel.
#
#   qsub -I -P CFP05-SF-002 -q auto -l select=1:ngpus=1 -l walltime=02:00:00   # get a GPU node
#   cd /scratch/e1583535/projects/robocolosseum-hosted
#   bash pbs_remote_inference/pbs/interactive.sh <robot> <family[:family...]> [tailscale|ngrok]
#   e.g. bash pbs_remote_inference/pbs/interactive.sh so101 molmoact2:groot tailscale
#
# Prints READY per model when it answers /healthz. Ctrl-C stops all servers and the tunnel
# (and logs the tailnet node out). Logs: logs/pbs_remote_inference/<robot>/<family>/.
set -e
cd "$(dirname "$0")/../.."    # repo root
ROBOT=${1:?usage: interactive.sh <robot> <family[:family...]> [tailscale|ngrok]}
MODELS=${2:?usage: interactive.sh <robot> <family[:family...]> [tailscale|ngrok]}
TUNNEL_KIND=${3:-}
export PBS_JOBID=${PBS_JOBID:-interactive-$$}
nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: no GPU here. Run this inside an interactive GPU job (qsub -I ...), not on the login node." >&2; exit 1; }

if [ -d /.singularity.d ] || [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
    echo "ERROR: you are inside a Singularity container. Type 'exit' once to return to the GPU node's shell, then rerun." >&2
    exit 1
fi
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh 2> /dev/null || true
module load singularity 2> /dev/null || true
command -v singularity > /dev/null || { echo "ERROR: singularity not found; run 'module load singularity' first." >&2; exit 1; }

if [ "$TUNNEL_KIND" = tailscale ]; then
    EMBODIMENT=$ROBOT TS_HOSTNAME=${TS_HOSTNAME:-colosseum-$ROBOT}
    source pbs_remote_inference/pbs/tailscale.sh    # joins the tailnet; logs out on exit
    export TUNNEL=""                                # tailnet covers all ports
else
    export TUNNEL=$TUNNEL_KIND                      # ngrok: one tunnel per server (serve.sh)
fi

# Each server in its own process group, so Ctrl-C here can stop it with everything it started.
PIDS=()
PORT=9101
for FAMILY in ${MODELS//:/ }; do
    setsid bash pbs_remote_inference/pbs/serve.sh "$FAMILY" "$ROBOT" "$PORT" &
    PIDS+=($!)
    PORT=$((PORT + 1))
done
stop() {
    echo "Stopping servers..."
    for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2> /dev/null || true; done
    wait 2> /dev/null || true
    exit 0          # runs tailscale.sh's EXIT trap (logout) when the tailnet is used
}
trap stop INT TERM

PORT=9101; i=0; OK=0
for FAMILY in ${MODELS//:/ }; do
    while ! curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" > /dev/null; do
        if ! kill -0 "${PIDS[$i]}" 2> /dev/null; then   # this family's server exited
            echo "FAILED $FAMILY/$ROBOT on port $PORT: see logs/pbs_remote_inference/$ROBOT/$FAMILY/stderr.$PBS_JOBID.log" >&2
            break
        fi
        sleep 3
    done
    curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" > /dev/null && { echo "READY $FAMILY/$ROBOT on 127.0.0.1:$PORT"; OK=$((OK + 1)); }
    PORT=$((PORT + 1)); i=$((i + 1))
done
[ $OK -gt 0 ] || { echo "ERROR: no model started." >&2; stop; }
[ "$TUNNEL_KIND" = tailscale ] && echo "Policy Server side: http://$TS_HOSTNAME:9101, :9102, ... (via the local relay)"
echo "All models up. Ctrl-C to stop."
wait
