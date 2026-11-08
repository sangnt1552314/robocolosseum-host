#!/bin/bash
# Start one colosseum_server.py for <family> on <robot> at 127.0.0.1:<port>, inside that
# family's container + venv. Run from the repo root on a GPU node after
# `module load singularity`. Used by <family>.pbs (one model) and cohost.pbs (several
# models sharing one GPU). Blocks until the server exits.
#
#   bash pbs_remote_inference/pbs/serve.sh <family> <robot> <port>
set -e
FAMILY=$1 ROBOT=$2 PORT=$3

case "$FAMILY" in
    molmoact2) IMAGE=pytorch_2.6.0_cuda_12.8.sif;            VENV=robocolosseum ;;   # transformers 5.16
    pi05)      IMAGE=pytorch_2.7.0_cuda_12.8.sif;            VENV=colosseum-pi05 ;;  # LeRobot b6ec006 + hqfang/pi05-so100_101 patch (projects/colosseum-deps)
    groot)     IMAGE=pytorch_2.7.0_cuda_12.8.sif;            VENV=lerobot-cu128 ;;   # editable LeRobot v0.6.1 (projects/lerobot), used read-only
    lingbot)   IMAGE=pytorch_2.6.0_cuda_12.8.sif;            VENV=lingbot-vla-v2 ;;  # projects/lingbot-vla-v2 @ 4144e9c
    g05)       IMAGE=pytorch_2.3.0_cuda_12.4_ngc_24.04.sif;  VENV=g05 ;;             # projects/colosseum-deps/GalaxeaVLA @ 89f2322
    *) echo "ERROR: unknown family '$FAMILY' (molmoact2, pi05, groot, lingbot, g05)" >&2; exit 1 ;;
esac
CONFIG="pbs_remote_inference/configs/$ROBOT/$FAMILY.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG" >&2; exit 1; }

LOG_DIR="./logs/pbs_remote_inference/$ROBOT/$FAMILY"
LOG="$LOG_DIR/%s.${PBS_JOBID:-local}.log"
mkdir -p "$LOG_DIR"
echo "$FAMILY/$ROBOT: $(hostname) 127.0.0.1:$PORT, logs in $LOG_DIR/*.${PBS_JOBID:-local}.log"

# Optional public HTTPS URL for this port. /act has no auth: give the URL only to the
# Policy Server operator. The URL is printed and written to endpoint.<jobid>.log.
TUNNEL_LOG="$(printf "$LOG" tunnel)"
case "${TUNNEL:-}" in
    ngrok)
        # Connects out on TCP 443 (works on Hopper). Account token NGROK_AUTHTOKEN in .env.
        # Basic auth is mandatory: NGROK_BASIC_AUTH=user:password[,user2:password2...] in .env
        # (one per Policy Server operator, max 10, passwords 8-128 chars); others get 401.
        # Free plan: one dev domain (one model at a time) and 1 GB/month (~400 so101 requests).
        CREDS=$(grep -E '^NGROK_BASIC_AUTH=' pbs_remote_inference/.env | cut -d= -f2-)
        [ -n "$CREDS" ] || { echo "ERROR: NGROK_BASIC_AUTH missing from pbs_remote_inference/.env" >&2; exit 1; }
        POLICY=$(mktemp /scratch/e1583535/tmp/ngrok-policy.XXXXXX.yml)   # mktemp: mode 600
        { echo "on_http_request:"; echo "  - actions:"; echo "      - type: basic-auth"
          echo "        config:"; echo "          credentials:"
          tr ',' '\n' <<< "$CREDS" | sed 's/.*/            - "&"/'; } > "$POLICY"
        NGROK_AUTHTOKEN=$(grep -E '^NGROK_AUTHTOKEN=' pbs_remote_inference/.env | cut -d= -f2-) \
            /home/svu/e1583535/bin/ngrok http "127.0.0.1:$PORT" --traffic-policy-file "$POLICY" \
            --log=stdout > "$TUNNEL_LOG" 2>&1 & ;;
    cloudflare)
        # Quick tunnel, no account. BLOCKED on Hopper GPU nodes: cloudflared needs TCP/UDP
        # 7844 to the Cloudflare edge. Kept for hosts where that port is open.
        /scratch/e1583535/tools/cloudflared/cloudflared tunnel --no-autoupdate --protocol http2 \
            --url "http://127.0.0.1:$PORT" > "$TUNNEL_LOG" 2>&1 & ;;
esac
if [ -n "${TUNNEL:-}" ]; then
    trap "kill $! 2>/dev/null; rm -f ${POLICY:-}" EXIT
    ( for _ in $(seq 90); do
          URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com|url=https://[^ ]+' "$TUNNEL_LOG" \
                | grep -v '//api\.' | head -1 | sed 's/^url=//')
          [ -n "$URL" ] && { echo "$FAMILY/$ROBOT endpoint: $URL" | tee "$(printf "$LOG" endpoint)"; break; }
          sleep 2
      done ) &
fi

singularity exec -e --nv \
    --env HF_HOME=/scratch/e1583535/cache \
    --env HF_HUB_OFFLINE=1 \
    --env TMPDIR=/scratch/e1583535/tmp \
    --env TEMP=/scratch/e1583535/tmp \
    --env TMP=/scratch/e1583535/tmp \
    --env FAMILY="$FAMILY" --env ROBOT="$ROBOT" --env PORT="$PORT" --env VENV="$VENV" --env CONFIG="$CONFIG" \
    "/app1/common/singularity-img/hopper/pytorch/$IMAGE" bash << 'EOS' > "$(printf "$LOG" stdout)" 2> "$(printf "$LOG" stderr)"
mkdir -p /scratch/e1583535/tmp
source /scratch/e1583535/virtualenvs/$VENV/bin/activate
if [ "$FAMILY" = lingbot ]; then
    # LingBot's fused MoE kernel is Triton, which links with -lcuda: point it at the
    # host driver injected by --nv (the image's compat libcuda.so.1 is unusable under -e).
    export TRANSFORMERS_OFFLINE=1 TRITON_LIBCUDA_PATH=/scratch/e1583535/tmp/cuda-stub-lib
    mkdir -p $TRITON_LIBCUDA_PATH
    [ -e $TRITON_LIBCUDA_PATH/libcuda.so ] || ln -sfn /.singularity.d/libs/libcuda.so.1 $TRITON_LIBCUDA_PATH/libcuda.so
fi
echo "Starting colosseum_server at $(date) on host $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"
python pbs_remote_inference/colosseum_server.py --robot "$ROBOT" --config "$CONFIG" --port "$PORT"
echo "colosseum_server ended at $(date)"
EOS
