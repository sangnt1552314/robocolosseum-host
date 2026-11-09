#!/bin/bash
# Start one colosseum_server.py for <family> on <robot> at 127.0.0.1:<port>, from that family's
# virtualenv. Run on a GPU node. Used by <family>.pbs (one model), cohost.pbs and interactive.sh
# (several models sharing one GPU). Blocks until the server exits.
#
#   bash nscc_pbs_remote_inference/pbs/serve.sh <family> <robot> <port>
set -euo pipefail
FAMILY=${1:?usage: serve.sh <family> <robot> <port>}
ROBOT=${2:?usage: serve.sh <family> <robot> <port>}
PORT=${3:?usage: serve.sh <family> <robot> <port>}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$HOST_ROOT"

case "$FAMILY" in
    molmoact2) VENV=$LEROBOT_VENV ;;   # transformers 5.5.4 (the Hopper venv had 5.16)
    groot)     VENV=$LEROBOT_VENV ;;   # LeRobot v0.6.1 (projects/lerobot)
    pi05)      VENV=$PI05_VENV ;;      # LeRobot b6ec006 + hqfang/pi05-so100_101 patch (setup.sh pi05)
    lingbot)   VENV=$LINGBOT_VENV ;;   # projects/lingbot-vla-v2 @ be969b8
    g05)       VENV=$G05_VENV ;;       # projects/GalaxeaVLA @ 89f2322
    *) echo "ERROR: unknown family '$FAMILY' (molmoact2, pi05, groot, lingbot, g05)" >&2; exit 1 ;;
esac
[ -x "$VENV/bin/python" ] || { echo "ERROR: no venv at $VENV (see pbs/setup.sh)" >&2; exit 1; }
CONFIG="$INFER_DIR/configs/$ROBOT/$FAMILY.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG" >&2; exit 1; }

JOB=${PBS_JOBID:-local}
LOG_DIR="$LOG_ROOT/$ROBOT/$FAMILY"
mkdir -p "$LOG_DIR"
echo "$FAMILY/$ROBOT: $(hostname) 127.0.0.1:$PORT, logs in $LOG_DIR/*.$JOB.log"

# Optional public HTTPS URL for this port (TUNNEL=tailscale is handled by the job script instead).
# /act has no auth of its own: give the URL only to the Policy Server operator.
TUNNEL_LOG="$LOG_DIR/tunnel.$JOB.log"
TUNNEL_PID=""
POLICY=""
case "${TUNNEL:-}" in
    ngrok)
        # Needs NGROK_AUTHTOKEN and NGROK_BASIC_AUTH=user:password[,user2:password2...] in .env.
        NGROK_BIN="$TOOLS_ROOT/ngrok/ngrok"
        [ -x "$NGROK_BIN" ] || { echo "ERROR: put the ngrok binary at $NGROK_BIN" >&2; exit 1; }
        CREDS=$(grep -E '^NGROK_BASIC_AUTH=' "$INFER_ENV_FILE" 2>/dev/null | cut -d= -f2- || true)
        [ -n "$CREDS" ] || { echo "ERROR: NGROK_BASIC_AUTH missing from $INFER_ENV_FILE" >&2; exit 1; }
        POLICY=$(mktemp "$TMPDIR/ngrok-policy.XXXXXX.yml")   # mktemp: mode 600
        { echo "on_http_request:"; echo "  - actions:"; echo "      - type: basic-auth"
          echo "        config:"; echo "          credentials:"
          tr ',' '\n' <<< "$CREDS" | sed 's/.*/            - "&"/'; } > "$POLICY"
        NGROK_AUTHTOKEN=$(grep -E '^NGROK_AUTHTOKEN=' "$INFER_ENV_FILE" | cut -d= -f2-) \
            "$NGROK_BIN" http "127.0.0.1:$PORT" --traffic-policy-file "$POLICY" \
            --log=stdout > "$TUNNEL_LOG" 2>&1 &
        TUNNEL_PID=$! ;;
    cloudflare)
        # Quick tunnel, no account. Needs TCP/UDP 7844 to the Cloudflare edge (not tested on NSCC).
        CLOUDFLARED_BIN="$TOOLS_ROOT/cloudflared/cloudflared"
        [ -x "$CLOUDFLARED_BIN" ] || { echo "ERROR: put the cloudflared binary at $CLOUDFLARED_BIN" >&2; exit 1; }
        "$CLOUDFLARED_BIN" tunnel --no-autoupdate --protocol http2 \
            --url "http://127.0.0.1:$PORT" > "$TUNNEL_LOG" 2>&1 &
        TUNNEL_PID=$! ;;
esac
if [ -n "$TUNNEL_PID" ]; then
    trap 'kill "$TUNNEL_PID" 2>/dev/null || true; rm -f "$POLICY"' EXIT
    ( for _ in $(seq 90); do
          URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com|url=https://[^ ]+' "$TUNNEL_LOG" 2>/dev/null \
                | grep -v '//api\.' | head -1 | sed 's/^url=//' || true)
          if [ -n "$URL" ]; then
              echo "$FAMILY/$ROBOT endpoint: $URL" | tee "$LOG_DIR/endpoint.$JOB.log"
              break
          fi
          sleep 2
      done ) &
fi

# Jobs run offline from the local snapshots named in the configs.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
{
    echo "Starting colosseum_server at $(date) on host $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-all}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"
    "$VENV/bin/python" "$INFER_DIR/colosseum_server.py" --robot "$ROBOT" --config "$CONFIG" --port "$PORT"
    echo "colosseum_server ended at $(date)"
} > "$LOG_DIR/stdout.$JOB.log" 2> "$LOG_DIR/stderr.$JOB.log"
