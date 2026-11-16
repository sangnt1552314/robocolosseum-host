#!/usr/bin/env bash
# Run an evaluation on the laptop with the models served from NSCC: the five relays, the Policy
# Server, then the web GUI. Ctrl-C stops everything. Start the robot's NSCC job first
# (TUNNEL=tailscale). Put this script in the folder that holds robocolosseum-client and
# robocolosseum-policy-server (e.g. ~/Workspace). It also waits for the job's SAM3 server, which
# needs no relay: http://<TS_HOST>:9106 (so101, used by the -capx LLMs) or :9116 (yam, unused until
# CaP-X supports yam).
#
#   bash ~/Workspace/run_nscc_eval.sh -e so101
#   bash ~/Workspace/run_nscc_eval.sh -e yam --offline
#
# Options:
#   -e, --embodiment so101|yam   robot on this laptop (required)
#   --offline                    no Router, nothing uploaded (testing only)
#   --policy-dir DIR             Policy Server checkout (default: <script folder>/robocolosseum-policy-server)
#   --client-dir DIR             client checkout (default: <script folder>/robocolosseum-client)
#   --wait SECONDS               how long to wait for the models (default: 300)
#   --no-sam3                    job started without sam3: don't wait for it
# Environment: POLICY_ENV (default colosseum-policy), CLIENT_ENV (default colosseum-client),
#   TS_HOST (default colosseum-<embodiment>).
# Logs: robocolosseum-client/local/logs/nscc-<embodiment>/
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
healthy() {
    local reply
    reply=$(curl -s -m 5 "http://127.0.0.1:$1/healthz" 2>/dev/null || true)
    case "$reply" in *'"ok"'*) return 0 ;; *) return 1 ;; esac
}
sam3_up() { curl -sf -m 5 -o /dev/null "$SAM3_URL/openapi.json"; }   # SAM3 has no /healthz

EMBODIMENT=""
OFFLINE=0
WAIT=300
USE_SAM3=1
POLICY_DIR="$SCRIPT_DIR/robocolosseum-policy-server"
CLIENT_DIR="$SCRIPT_DIR/robocolosseum-client"
while [ $# -gt 0 ]; do
    case "$1" in
        -e|--embodiment|--policy-dir|--client-dir|--wait)
            [ -n "${2:-}" ] || die "$1 needs a value (see --help)" ;;
    esac
    case "$1" in
        -e|--embodiment) EMBODIMENT=$2; shift 2 ;;
        --offline) OFFLINE=1; shift ;;
        --policy-dir) POLICY_DIR=$2; shift 2 ;;
        --client-dir) CLIENT_DIR=$2; shift 2 ;;
        --wait) WAIT=$2; shift 2 ;;
        --no-sam3) USE_SAM3=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

# Ports and model order match the NSCC job's MODELS (robocolosseum-host/nscc_pbs_remote_inference/README.md).
# Both jobs run SAM3 as a sixth family on the port after the five VLAs; it needs no relay.
case "$EMBODIMENT" in
    so101) REMOTE_BASE=9101; LOCAL_BASE=9201; JOB_MODELS=molmoact2:pi05:groot:g05:lingbot:sam3
           MODELS=(MolmoAct2 pi0.5 "GR00T N1.7" G05 LingBot) ;;
    yam)   REMOTE_BASE=9111; LOCAL_BASE=9211; JOB_MODELS=molmoact2:pi05:groot:lingbot:g05:sam3
           MODELS=(MolmoAct2 pi0.5 "GR00T N1.7" LingBot G05) ;;
    *) usage >&2; die "-e must be so101 or yam" ;;
esac
case "$WAIT" in ''|*[!0-9]*) die "--wait must be a number of seconds" ;; esac
if [ "$USE_SAM3" -eq 0 ]; then JOB_MODELS=${JOB_MODELS%:sam3}; fi

TS_HOST=${TS_HOST:-colosseum-$EMBODIMENT}
SAM3_URL=http://$TS_HOST:$((REMOTE_BASE + 5))
POLICY_ENV=${POLICY_ENV:-colosseum-policy}
CLIENT_ENV=${CLIENT_ENV:-colosseum-client}
SERVER_CONFIG=configs/local-runtime-$EMBODIMENT.yaml
ROBOT_CONFIG=configs/robot.$EMBODIMENT.yaml

# ---------- checks ----------
POLICY_DIR=$(cd "$POLICY_DIR" 2>/dev/null && pwd) || die "no $POLICY_DIR; pass --policy-dir"
CLIENT_DIR=$(cd "$CLIENT_DIR" 2>/dev/null && pwd) || die "no $CLIENT_DIR; pass --client-dir"
RELAY=$POLICY_DIR/scripts/policy_server_relay.py
LOG_DIR=$CLIENT_DIR/local/logs/nscc-$EMBODIMENT
[ -f "$RELAY" ] || die "missing $RELAY; update robocolosseum-policy-server (git pull)"
[ -f "$POLICY_DIR/$SERVER_CONFIG" ] || die "missing $POLICY_DIR/$SERVER_CONFIG"
[ -f "$CLIENT_DIR/$ROBOT_CONFIG" ] || die "missing $CLIENT_DIR/$ROBOT_CONFIG"
[ -f "$CLIENT_DIR/scripts/eval_gui.py" ] || die "missing $CLIENT_DIR/scripts/eval_gui.py"
command -v python3 >/dev/null || die "python3 not found (the relays need it)"
command -v curl >/dev/null || die "curl not found"

SERVER_PORT=$(awk '$1 == "port:" { print $2; exit }' "$POLICY_DIR/$SERVER_CONFIG")
SERVER_PORT=${SERVER_PORT:-8000}
grep -Eq "^policy_server_url:[[:space:]]*[\"']?ws://127\.0\.0\.1:$SERVER_PORT" "$CLIENT_DIR/$ROBOT_CONFIG" \
    || die "$CLIENT_DIR/$ROBOT_CONFIG: policy_server_url must be ws://127.0.0.1:$SERVER_PORT"

# The -capx LLMs (SO-101 only) read capx.sam3_url from the Policy Server config and fail a session
# if it is closed.
if [ "$USE_SAM3" -eq 1 ] && [ "$EMBODIMENT" = so101 ]; then
    cfg_sam3=$(awk '$1 == "sam3_url:" { print $2; exit }' "$POLICY_DIR/$SERVER_CONFIG" | tr -d "\"'")
    if [ -z "$cfg_sam3" ]; then
        echo "NOTE: $SERVER_CONFIG has no capx.sam3_url; -capx models need capx.sam3_url: $SAM3_URL"
    elif [ "${cfg_sam3%/}" != "$SAM3_URL" ]; then
        echo "WARNING: $SERVER_CONFIG has capx.sam3_url: $cfg_sam3, but the NSCC SAM3 server is $SAM3_URL"
    fi
fi

for i in 0 1 2 3 4; do
    port_open $((LOCAL_BASE + i)) \
        && die "port $((LOCAL_BASE + i)) is in use; stop old relays with: pkill -f policy_server_relay.py"
done
port_open "$SERVER_PORT" && die "port $SERVER_PORT is in use; stop the Policy Server that is already running"

if command -v tailscale >/dev/null; then
    ts_line=$(tailscale status 2>/dev/null | awk -v h="$TS_HOST" '$2 == h' || true)
    [ -n "$ts_line" ] || die "$TS_HOST is not on your tailnet. Start the $EMBODIMENT job with TUNNEL=tailscale" \
        "and MODELS=$JOB_MODELS, then check 'tailscale status'."
    case "$ts_line" in *offline*) die "$TS_HOST is offline in 'tailscale status'; is the job still running?" ;; esac
fi

CONDA_BIN=""
for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" \
         "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then CONDA_BIN=$c; break; fi
done
[ -n "$CONDA_BIN" ] || die "conda not found; set CONDA_EXE=/path/to/bin/conda"
# shellcheck disable=SC1091
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
for env in "$POLICY_ENV" "$CLIENT_ENV"; do
    conda env list | awk -v e="$env" '$1 == e { found = 1 } END { exit !found }' \
        || die "conda env '$env' not found"
done

# ---------- cleanup on exit ----------
RELAY_PIDS=()
SERVER_PID=""
cleanup() {
    trap - EXIT INT TERM
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping the Policy Server..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    if [ ${#RELAY_PIDS[@]} -gt 0 ]; then
        echo "Stopping the relays..."
        kill "${RELAY_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p "$LOG_DIR"

# ---------- 1. relays ----------
echo "[1/3] Relays: $TS_HOST:$REMOTE_BASE-$((REMOTE_BASE + 4)) -> 127.0.0.1:$LOCAL_BASE-$((LOCAL_BASE + 4))"
for i in 0 1 2 3 4; do
    python3 "$RELAY" --remote "http://$TS_HOST:$((REMOTE_BASE + i))" \
        --port $((LOCAL_BASE + i)) > "$LOG_DIR/relay-$((LOCAL_BASE + i)).log" 2>&1 &
    RELAY_PIDS+=($!)
done

echo "      Waiting up to ${WAIT}s for the models (Ctrl-C to abort)..."
deadline=$((SECONDS + WAIT))
next_report=$((SECONDS + 15))
SAM3_READY=0
while :; do
    READY=()
    for i in 0 1 2 3 4; do
        if healthy $((LOCAL_BASE + i)); then READY[i]=1; else READY[i]=0; fi
    done
    n_ready=$((READY[0] + READY[1] + READY[2] + READY[3] + READY[4]))
    if [ "$USE_SAM3" -eq 1 ] && [ "$SAM3_READY" -eq 0 ] && sam3_up; then SAM3_READY=1; fi
    [ "$n_ready" -eq 5 ] && { [ "$USE_SAM3" -eq 0 ] || [ "$SAM3_READY" -eq 1 ]; } && break
    for pid in "${RELAY_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || die "a relay exited; see $LOG_DIR/relay-*.log"
    done
    [ "$SECONDS" -ge "$deadline" ] && break
    if [ "$SECONDS" -ge "$next_report" ]; then
        if [ "$USE_SAM3" -eq 1 ]; then
            echo "      $n_ready/5 ready, SAM3 $([ "$SAM3_READY" -eq 1 ] && echo ready || echo 'not yet')..."
        else
            echo "      $n_ready/5 ready..."
        fi
        next_report=$((SECONDS + 15))
    fi
    sleep 3
done
for i in 0 1 2 3 4; do
    status="ready"; [ "${READY[i]}" -eq 1 ] || status="NOT READY"
    printf '      %-11s http://127.0.0.1:%s  %s\n' "${MODELS[i]}" $((LOCAL_BASE + i)) "$status"
done
if [ "$USE_SAM3" -eq 1 ]; then
    status="ready"; [ "$SAM3_READY" -eq 1 ] || status="NOT READY"
    printf '      %-11s %s  %s (no relay)\n' SAM3 "$SAM3_URL" "$status"
fi
[ "$n_ready" -gt 0 ] || die "no model answered through $TS_HOST. Check the job (qstat -u \$USER) and" \
    "that it runs with TUNNEL=tailscale and MODELS=$JOB_MODELS."
[ "$n_ready" -eq 5 ] || echo "WARNING: continuing without the models marked NOT READY."
if [ "$USE_SAM3" -eq 1 ] && [ "$SAM3_READY" -eq 0 ]; then
    impact=""; [ "$EMBODIMENT" = so101 ] && impact=" the -capx models will fail."
    echo "WARNING: SAM3 is not answering at $SAM3_URL;$impact Check" \
        "$EMBODIMENT/sam3/stderr.<jobid>.log on NSCC, or pass --no-sam3 if the job runs without it."
fi

# ---------- 2. Policy Server ----------
echo "[2/3] Policy Server: $POLICY_DIR/$SERVER_CONFIG"
( conda activate "$POLICY_ENV" && cd "$POLICY_DIR" && exec colosseum-policy-local --config "$SERVER_CONFIG" ) \
    > "$LOG_DIR/policy-server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 120))
until port_open "$SERVER_PORT"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -n 20 "$LOG_DIR/policy-server.log" >&2
        die "the Policy Server exited; see $LOG_DIR/policy-server.log"
    fi
    [ "$SECONDS" -lt "$deadline" ] || die "the Policy Server did not start in 120 s; see $LOG_DIR/policy-server.log"
    sleep 1
done
echo "      Running on ws://127.0.0.1:$SERVER_PORT"

# ---------- 3. web GUI ----------
GUI_ARGS=(--config "$ROBOT_CONFIG" --server-dir "$POLICY_DIR" --server-config "$SERVER_CONFIG")
if [ "$OFFLINE" -eq 1 ]; then GUI_ARGS+=(--offline); fi
echo "[3/3] Web GUI: open http://127.0.0.1:8770 (Ctrl-C here stops everything; logs in $LOG_DIR)"
status=0
( conda activate "$CLIENT_ENV" && cd "$CLIENT_DIR" && exec python scripts/eval_gui.py "${GUI_ARGS[@]}" ) || status=$?
exit "$status"
