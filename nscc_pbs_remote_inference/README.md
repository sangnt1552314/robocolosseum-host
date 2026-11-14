# nscc_pbs_remote_inference

Serve VLA models on NSCC ASPIRE 2B H200 GPUs for
[Colosseum Policy Server](https://github.com/frodobots-org/colosseum-policy-server).
This is the NSCC version of [`pbs_remote_inference`](../pbs_remote_inference/README.md). Each
embodiment runs as one job with 5 models on 5 GPUs, one model per GPU, reached over Tailscale. There are no containers: each
model family runs from its own uv virtualenv under `/scratch/users/nus/duanj1/tansang`.
Checkpoints are in `/data/projects/51001013/tansang/cache`, because scratch has a file-count limit.

## First-time setup (login node)

```bash
cd /scratch/users/nus/duanj1/tansang/projects/robocolosseum-host
bash nscc_pbs_remote_inference/pbs/setup.sh all     # checkpoints, pi05 venv, Tailscale binaries, check
cp nscc_pbs_remote_inference/.env.example nscc_pbs_remote_inference/.env   # then set TS_AUTHKEY
```

`.env` is git-ignored and needs a reusable + ephemeral Tailscale auth key. Jobs without
`TUNNEL=tailscale` don't need it. Checkpoints must be downloaded first, because jobs run offline.

## 1. Start the models

**Batch job** (recommended for evaluations). Each job uses 5 GPUs. The default walltime is 8 h, which runs in `g1` (24 h maximum);
`-l walltime=02:00:00` or less runs in `gdev`.

```bash
qsub -v EMBODIMENT=so101,MODELS=molmoact2:pi05:groot:g05:lingbot,TUNNEL=tailscale nscc_pbs_remote_inference/pbs/cohost.pbs
qsub -v EMBODIMENT=yam,MODELS=molmoact2:pi05:groot:lingbot:g05,TUNNEL=tailscale nscc_pbs_remote_inference/pbs/cohost.pbs
```

**Interactive job** (quick tests). It ends when your terminal closes, so use `tmux` for long runs:

```bash
qsub -I -P 11010001 -q normal -l select=1:ncpus=60:ngpus=5:mem=900gb -l walltime=02:00:00
cd /scratch/users/nus/duanj1/tansang/projects/robocolosseum-host
bash nscc_pbs_remote_inference/pbs/interactive.sh so101 molmoact2:pi05:groot:g05:lingbot tailscale
```

The script prints `READY <model>` per model; Ctrl-C stops everything.

**Check / stop a batch job:**

```bash
qstat -u $USER                                                              # R = running
cat /data/projects/51001013/tansang/logs/nscc_pbs_remote_inference/tailscale/status.<jobid>.txt             # tailnet node is up
grep -h Serving /data/projects/51001013/tansang/logs/nscc_pbs_remote_inference/so101/*/stdout.<jobid>.log   # one line per ready model
qdel <jobid>                                                                # stop
```

Logs are in `/data/projects/51001013/tansang/logs/nscc_pbs_remote_inference/<robot>/<model>/`, and the PBS output is in `/data/projects/51001013/tansang/logs/pbs/`.

## 2. Connect the Policy Server machine

On the Policy Server machine (once): install Tailscale and log in to the same tailnet. The relay is
`nscc_pbs_remote_inference/policy_server_relay.py`; `robocolosseum-policy-server` ships the same
script as `scripts/policy_server_relay.py`.

```bash
tailscale status          # colosseum-so101 and colosseum-yam should be listed
# relays: the Policy Server only accepts http://127.0.0.1:<port> endpoints
for i in 1 2 3 4 5; do python3 policy_server_relay.py --remote http://colosseum-so101:910$i --port 920$i & done
for i in 1 2 3 4 5; do python3 policy_server_relay.py --remote http://colosseum-yam:911$i   --port 921$i & done
```

Unlike Hopper, yam listens on ports 9111-9115, because `g1` jobs can share a node.

Endpoints for the Policy Server config:

| so101 endpoint | model | yam endpoint | model |
|---|---|---|---|
| http://127.0.0.1:9201 | molmoact2 | http://127.0.0.1:9211 | molmoact2 |
| http://127.0.0.1:9202 | pi05 | http://127.0.0.1:9212 | pi05 |
| http://127.0.0.1:9203 | groot | http://127.0.0.1:9213 | groot |
| http://127.0.0.1:9204 | g05 | http://127.0.0.1:9214 | lingbot |
| http://127.0.0.1:9205 | lingbot | http://127.0.0.1:9215 | g05 |

The Hugging Face url, revision, `max_horizon` and `adapter` for each entry are in
[DETAILS.md](DETAILS.md#models). Leave `joint_frame` unset for so101.

## 3. Run an evaluation from the laptop

Start the robot's job first (section 1) with `TUNNEL=tailscale` and the `MODELS` order shown there;
the relays below assume that order. `tailscale status` on the laptop should list `colosseum-so101`
or `colosseum-yam`. Then open three terminals, in this order.

**Terminal 1: relays**, from `robocolosseum-policy-server`.

so101:

```bash
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9101 --port 9201 &   # MolmoAct2
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9102 --port 9202 &   # pi0.5
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9103 --port 9203 &   # GR00T N1.7
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9104 --port 9204 &   # G05
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9105 --port 9205     # LingBot
```

yam:

```bash
python3 scripts/policy_server_relay.py --remote http://colosseum-yam:9111 --port 9211 &   # MolmoAct2
python3 scripts/policy_server_relay.py --remote http://colosseum-yam:9112 --port 9212 &   # pi0.5
python3 scripts/policy_server_relay.py --remote http://colosseum-yam:9113 --port 9213 &   # GR00T N1.7
python3 scripts/policy_server_relay.py --remote http://colosseum-yam:9114 --port 9214 &   # LingBot
python3 scripts/policy_server_relay.py --remote http://colosseum-yam:9115 --port 9215     # G05
```

Ctrl-C stops only the last relay; the other four keep running in the background. Stop them all
with `pkill -f policy_server_relay.py`.

**Terminal 2: Policy Server**, from `robocolosseum-policy-server`:

```bash
conda activate colosseum-policy
colosseum-policy-local --config configs/local-runtime-so101.yaml   # so101
colosseum-policy-local --config configs/local-runtime-yam.yaml     # yam
```

**Terminal 3: web GUI**, from `colosseum-client`, then open http://127.0.0.1:8770:

```bash
conda activate colosseum-client
# so101
python scripts/eval_gui.py --config configs/robot.so101.yaml --server-config configs/local-runtime-so101.yaml --offline # testing only
python scripts/eval_gui.py --config configs/robot.so101.yaml --server-config configs/local-runtime-so101.yaml
# yam
python scripts/eval_gui.py --config configs/robot.yam.yaml --server-config configs/local-runtime-yam.yaml --offline # testing only
python scripts/eval_gui.py --config configs/robot.yam.yaml --server-config configs/local-runtime-yam.yaml
```

Run one line for your robot. `--offline`: no Router; two local policies picked at random, nothing
uploaded. Without it, the Router assigns the models and results are submitted to it. The robot
config's `policy_server_url` must be `ws://127.0.0.1:8000`. The GUI detects the Policy Server from
terminal 2 and uses it instead of starting its own.

### If the GUI reports "No permission for /dev/ttyACM0"

This happens on Linux; macOS serial ports (`/dev/tty.usbmodem*`) need no fix. The GUI suggests
`scripts/fix_serial_permissions.sh`, which the client repository does not ship. Save the script
below as `fix_serial_permissions.sh` and run it with the arm's port:

```bash
bash fix_serial_permissions.sh /dev/ttyACM0
```

```bash
#!/usr/bin/env bash
# Let the current user open the SO101 arm's serial port.
#   bash scripts/fix_serial_permissions.sh [PORT]   (default: /dev/ttyACM0)
# Grants access now (ACL on the device) and permanently (dialout group, from your next login).
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
USER_NAME="${SUDO_USER:-$USER}"

if [ ! -e "$PORT" ]; then
  echo "No device at $PORT. Plug in the arm, or pass the port: $0 /dev/ttyACM1" >&2
  exit 1
fi
DEV="$(readlink -f "$PORT")"

GROUP="$(stat -c %G "$DEV")"
if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$GROUP"; then
  echo "Adding $USER_NAME to group $GROUP (takes effect at next login)…"
  sudo usermod -aG "$GROUP" "$USER_NAME"
fi

echo "Granting $USER_NAME read/write on $DEV now…"
if command -v setfacl >/dev/null; then
  sudo setfacl -m "u:$USER_NAME:rw" "$DEV"
else
  sudo chmod a+rw "$DEV"
fi

if [ -r "$DEV" ] && [ -w "$DEV" ]; then
  echo "OK: $PORT is readable and writable. Re-run the check in the GUI."
else
  echo "Still no access to $DEV; log out and back in, then try again." >&2
  exit 1
fi
```

It gives you access immediately (an ACL on the device) and adds you to the device's group
(usually `dialout`) for future logins. Then re-run the check in the GUI.

Everything else (models, environments, measurements, other tunnels, HTTP contract, adding a model)
is in [DETAILS.md](DETAILS.md).
