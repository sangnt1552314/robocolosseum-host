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


# A job killed before it logs out (walltime, qdel, node failure) leaves an offline device holding
# $TS_HOSTNAME, and the next job then joins as "$TS_HOSTNAME-1". With TS_API_KEY in .env (Tailscale
# admin console > Settings > Keys > API access token), delete such OFFLINE devices before joining.
# Online devices (e.g. another job still running for this robot) are never touched.
TS_API_KEY=$(grep -E '^TS_API_KEY=' "$INFER_ENV_FILE" 2>/dev/null | cut -d= -f2- || true)
if [ -n "$TS_API_KEY" ]; then
    TS_API_KEY="$TS_API_KEY" TS_HOSTNAME="$TS_HOSTNAME" python3 -I - <<'PY' \
        || echo "WARNING: stale tailnet device cleanup failed; continuing" >&2
import json, os, urllib.request
key, host = os.environ["TS_API_KEY"], os.environ["TS_HOSTNAME"]
def call(method, path):
    req = urllib.request.Request("https://api.tailscale.com/api/v2" + path, method=method,
                                 headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()
for d in json.loads(call("GET", "/tailnet/-/devices"))["devices"]:
    if d.get("hostname") == host and d.get("connectedToControl") is False:
        call("DELETE", "/device/" + d["id"])
        print("tailnet: deleted stale offline device", d.get("name"), "(last seen", d.get("lastSeen"), ")")
PY
fi
unset TS_API_KEY

"$TS_BIN/tailscaled" --tun=userspace-networking --state=mem: --socket="$TS_SOCK" \
    > "$TS_LOG_DIR/tailscaled.${PBS_JOBID:-local}.log" 2>&1 &
# On exit, log out first: that deletes the ephemeral node at once, so the next job gets the
# same hostname instead of a "-1" suffix. PBS ends a job with SIGTERM, which skips the EXIT trap
# unless it is trapped too, so turn TERM/INT/HUP into a normal exit.
trap "$TS_BIN/tailscale --socket=$TS_SOCK logout 2>/dev/null; kill $! 2>/dev/null; rm -f $TS_SOCK" EXIT
trap 'exit 143' TERM INT HUP
sleep 2
"$TS_BIN/tailscale" --socket="$TS_SOCK" up --auth-key="$TS_AUTHKEY" --hostname="$TS_HOSTNAME" --timeout=90s || {
    echo "ERROR: could not join the tailnet; see $TS_LOG_DIR/tailscaled.${PBS_JOBID:-local}.log" >&2; exit 1; }
unset TS_AUTHKEY
TS_STATUS=$("$TS_BIN/tailscale" --socket="$TS_SOCK" status --peers=false | head -1)
echo "tailnet: $TS_STATUS" | tee "$TS_LOG_DIR/status.${PBS_JOBID:-local}.txt"
TS_NAME=$(awk '{print $2}' <<< "$TS_STATUS")
if [ -n "$TS_NAME" ] && [ "$TS_NAME" != "$TS_HOSTNAME" ]; then
    echo "WARNING: joined the tailnet as '$TS_NAME', not '$TS_HOSTNAME': an old device still holds" \
         "that name. Use http://$TS_NAME:<port>, or delete the offline '$TS_HOSTNAME' in the admin" \
         "console (or set TS_API_KEY in .env so the next job does it)." >&2
fi
echo "endpoints: http://${TS_NAME:-$TS_HOSTNAME}:<port> (ports below)"
