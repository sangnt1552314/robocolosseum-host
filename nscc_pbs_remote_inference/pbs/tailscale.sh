# Sourced by the job scripts when TUNNEL=tailscale (after env.sh). Joins your tailnet as $TS_HOSTNAME
# (userspace networking, no root, ephemeral), so tailnet devices such as the Policy Server machine
# reach this job's servers at http://$TS_HOSTNAME:<port>. Incoming tailnet connections are forwarded
# to 127.0.0.1:<port>, where the servers listen.
# Needs TS_AUTHKEY (reusable + ephemeral auth key) in nscc_pbs_remote_inference/.env.

TS_BIN="$TOOLS_ROOT/tailscale"
TS_LOG_DIR="$LOG_ROOT/tailscale"
TS_SOCK="$TMPDIR/tailscale-${PBS_JOBID:-local}.sock"
mkdir -p "$TS_LOG_DIR"
[ -x "$TS_BIN/tailscaled" ] || { echo "ERROR: no tailscaled in $TS_BIN; run 'bash nscc_pbs_remote_inference/pbs/setup.sh tailscale' on the login node." >&2; exit 1; }
TS_AUTHKEY=$(grep -E '^TS_AUTHKEY=' "$INFER_ENV_FILE" 2>/dev/null | cut -d= -f2- || true)
[ -n "$TS_AUTHKEY" ] || { echo "ERROR: TS_AUTHKEY missing from $INFER_ENV_FILE (see .env.example)" >&2; exit 1; }

"$TS_BIN/tailscaled" --tun=userspace-networking --state=mem: --socket="$TS_SOCK" \
    > "$TS_LOG_DIR/tailscaled.${PBS_JOBID:-local}.log" 2>&1 &
# On exit, log out first: that deletes the ephemeral node at once, so the next job gets the
# same hostname instead of a "-1" suffix.
trap "$TS_BIN/tailscale --socket=$TS_SOCK logout 2>/dev/null; kill $! 2>/dev/null; rm -f $TS_SOCK" EXIT
sleep 2
"$TS_BIN/tailscale" --socket="$TS_SOCK" up --auth-key="$TS_AUTHKEY" --hostname="$TS_HOSTNAME" --timeout=90s || {
    echo "ERROR: could not join the tailnet; see $TS_LOG_DIR/tailscaled.${PBS_JOBID:-local}.log" >&2; exit 1; }
unset TS_AUTHKEY
echo "tailnet: $("$TS_BIN/tailscale" --socket="$TS_SOCK" status --peers=false | head -1)" \
    | tee "$TS_LOG_DIR/status.${PBS_JOBID:-local}.txt"
echo "endpoints: http://$TS_HOSTNAME:<port> (ports below)"
