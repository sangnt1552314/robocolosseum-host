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
bash nscc_pbs_remote_inference/pbs/setup.sh sam3    # SAM3 for the -capx LLMs (see "SAM3" below)
cp nscc_pbs_remote_inference/.env.example nscc_pbs_remote_inference/.env   # then set TS_AUTHKEY
```

`.env` is git-ignored and needs a reusable + ephemeral Tailscale auth key. Jobs without
`TUNNEL=tailscale` don't need it. Checkpoints must be downloaded first, because jobs run offline.

## 1. Start the models

**Batch job** (recommended for evaluations). Each job uses 5 GPUs. The default walltime is 8 h, which runs in `g1` (24 h maximum);
`-l walltime=02:00:00` or less runs in `gdev`.

```bash
qsub -v EMBODIMENT=so101,MODELS=molmoact2:pi05:groot:g05:lingbot:sam3,TUNNEL=tailscale nscc_pbs_remote_inference/pbs/cohost.pbs
qsub -v EMBODIMENT=yam,MODELS=molmoact2:pi05:groot:lingbot:g05:sam3,TUNNEL=tailscale nscc_pbs_remote_inference/pbs/cohost.pbs
```

**Interactive job** (quick tests). It ends when your terminal closes, so use `tmux` for long runs:

```bash
qsub -I -P 11010001 -q normal -l select=1:ncpus=60:ngpus=5:mem=900gb -l walltime=02:00:00
cd /scratch/users/nus/duanj1/tansang/projects/robocolosseum-host
bash nscc_pbs_remote_inference/pbs/interactive.sh so101 molmoact2:pi05:groot:g05:lingbot:sam3 tailscale
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

As on Hopper, yam listens on ports 9111-9115, so an so101 and a yam job on the same node don't collide.

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

## SAM3 for CaP-X LLMs

The Policy Server's `-capx` models (CaP-X code-as-policy LLMs) call a SAM3 segmentation server
(`POST /segment`, `POST /segment_point`). The laptop has no GPU, so each job runs SAM3 as a sixth
family: the commands in section 1 end with `:sam3`. CaP-X supports only SO-101 for now, so the yam
job's SAM3 sits unused until it supports yam. Set it up once on the login node with
`setup.sh sam3`. This installs Meta's SAM3 code and downloads `sam3.pt` (3.45 GB), which means
accepting Meta's SAM License. modelscope.cn is unreachable from NSCC, so download `sam3.pt`
elsewhere and copy it to `/data/projects/51001013/tansang/cache/modelscope/models/facebook/sam3/sam3.pt`
first; `setup.sh sam3` then skips the download.

Keep `sam3` last, so the VLAs keep their ports and SAM3 gets the next one: 9106 for so101, 9116 for
yam. With 5 GPUs it shares GPU 0 with MolmoAct2; with 1 GPU all six share it. No relay is needed:
the Policy Server reaches SAM3 directly over Tailscale, so its so101 config needs
`capx.sam3_url: http://colosseum-so101:9106`.
To run without it, drop `:sam3` from `MODELS` and pass `--no-sam3` to `run_nscc_eval.sh`.

SAM3 has no `/healthz`. Check it from the laptop with:

```bash
curl -sf http://colosseum-so101:9106/openapi.json > /dev/null && echo SAM3 up   # yam: colosseum-yam:9116
```

On NSCC, its log is `/data/projects/51001013/tansang/logs/nscc_pbs_remote_inference/<robot>/sam3/stderr.<jobid>.log`
(`Uvicorn running on http://127.0.0.1:<port>` when ready).

## 3. Run an evaluation from the laptop

Start the robot's job first (section 1) with `TUNNEL=tailscale` and the `MODELS` order shown there;
the relays below assume that order. `tailscale status` on the laptop should list `colosseum-so101`
or `colosseum-yam`.

**One command.** Download the script once into `~/Workspace`, next to `robocolosseum-client` and
`robocolosseum-policy-server`:

```bash
curl -fsSL -o ~/Workspace/run_nscc_eval.sh https://raw.githubusercontent.com/sangnt1552314/robocolosseum-host/master/nscc_pbs_remote_inference/run_nscc_eval.sh
```

Then run it and open http://127.0.0.1:8770:

```bash
bash ~/Workspace/run_nscc_eval.sh -e so101   # or -e yam; add --offline for testing only
```

It does terminals 1-3 below: relays, wait for the models and SAM3 (port 9106 for so101, 9116 for
yam), Policy Server, then the GUI. For so101 it also warns when the Policy Server config's
`capx.sam3_url` is not `http://colosseum-so101:9106`. Ctrl-C stops everything. `--help` lists the
options. Logs go to `robocolosseum-client/local/logs/nscc-<robot>/`.

**Or by hand**, in three terminals, in this order.

**Terminal 1: relays**, from `robocolosseum-policy-server`.

so101:

```bash
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9101 --port 9201 &   # MolmoAct2
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9102 --port 9202 &   # pi0.5
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9103 --port 9203 &   # GR00T N1.7
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9104 --port 9204 &   # G05
python3 scripts/policy_server_relay.py --remote http://colosseum-so101:9105 --port 9205     # LingBot
# SAM3 (port 9106) needs no relay: capx.sam3_url points straight at http://colosseum-so101:9106
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
