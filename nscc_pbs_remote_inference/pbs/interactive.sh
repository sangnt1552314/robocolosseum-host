#!/bin/bash
# Serve models from an INTERACTIVE GPU job. Same layout as cohost.pbs: one colosseum_server.py per
# family, each on its own GPU, on consecutive ports from 127.0.0.1:9101 (so101), 9111 (yam) or 9121 (franka), plus an
# optional tunnel.
#
#   qsub -I -P 11010001 -q normal -l select=1:ncpus=60:ngpus=5:mem=900gb -l walltime=02:00:00
#   bash /scratch/users/nus/duanj1/tansang/projects/robocolosseum-host/nscc_pbs_remote_inference/pbs/interactive.sh \
#       <robot> <family[:family...]> [tailscale|ngrok]
#
# Prints READY per model when it answers /healthz. Ctrl-C stops all servers and the tunnel
# (and logs the tailnet node out). Logs: /data/projects/51001013/tansang/logs/nscc_pbs_remote_inference/<robot>/<family>/.
set -euo pipefail
ROBOT=${1:?usage: interactive.sh <robot> <family[:family...]> [tailscale|ngrok]}
MODELS=${2:?usage: interactive.sh <robot> <family[:family...]> [tailscale|ngrok]}
TUNNEL_KIND=${3:-}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$HOST_ROOT"
BASE_PORT=${BASE_PORT:-$(nscc_base_port "$ROBOT")}
export PBS_JOBID=${PBS_JOBID:-interactive-$$}
nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: no GPU here. Run this inside an interactive GPU job (qsub -I ...), not on the login node." >&2; exit 1; }

if [ "$TUNNEL_KIND" = tailscale ]; then
    TS_HOSTNAME=${TS_HOSTNAME:-colosseum-$ROBOT}
    source "$INFER_DIR/pbs/tailscale.sh"    # joins the tailnet; logs out on exit
    export TUNNEL=""                        # tailnet covers all ports
else
    export TUNNEL=$TUNNEL_KIND              # ngrok: one tunnel per server (serve.sh)
fi

# Each server in its own process group, so Ctrl-C here can stop it with everything it started.
mapfile -t GPUS < <(nscc_gpu_list)
FAMILIES=(${MODELS//:/ })
export OMP_NUM_THREADS=$(( ${NCPUS:-60} / ${#FAMILIES[@]} > 0 ? ${NCPUS:-60} / ${#FAMILIES[@]} : 1 ))
PIDS=()
PORT=$BASE_PORT
n=0
for FAMILY in "${FAMILIES[@]}"; do
    GPU=${GPUS[$((n % ${#GPUS[@]}))]}
    echo "$FAMILY -> port $PORT, GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU setsid bash "$INFER_DIR/pbs/serve.sh" "$FAMILY" "$ROBOT" "$PORT" &
    PIDS+=($!)
    PORT=$((PORT + 1)); n=$((n + 1))
done
stop() {
    echo "Stopping servers..."
    for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2> /dev/null || true; done
    wait 2> /dev/null || true
    exit 0          # runs tailscale.sh's EXIT trap (logout) when the tailnet is used
}
trap stop INT TERM

PORT=$BASE_PORT; i=0; OK=0
for FAMILY in ${MODELS//:/ }; do
    while ! curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" > /dev/null; do
        if ! kill -0 "${PIDS[$i]}" 2> /dev/null; then   # this family's server exited
            echo "FAILED $FAMILY/$ROBOT on port $PORT: see $LOG_ROOT/$ROBOT/$FAMILY/stderr.$PBS_JOBID.log" >&2
            break
        fi
        sleep 3
    done
    if curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" > /dev/null; then
        echo "READY $FAMILY/$ROBOT on 127.0.0.1:$PORT"
        OK=$((OK + 1))
    fi
    PORT=$((PORT + 1)); i=$((i + 1))
done
[ $OK -gt 0 ] || { echo "ERROR: no model started." >&2; stop; }
if [ "$TUNNEL_KIND" = tailscale ]; then
    echo "Policy Server side: http://$TS_HOSTNAME:$BASE_PORT, ... (via the local relay)"
fi
echo "All models up. Ctrl-C to stop."
wait
