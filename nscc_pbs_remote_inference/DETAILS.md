# nscc_pbs_remote_inference: details

Reference for [README.md](README.md): models, environments, measurements, tunnel options, HTTP contract.
The code, configs and contract come from `pbs_remote_inference` (Hopper). Latency and memory numbers below
were measured on Hopper and have not been re-measured on NSCC.

## Layout

```
nscc_pbs_remote_inference/
├── colosseum_server.py            # HTTP server (stdlib + numpy): GET /healthz, POST /act
├── configs/<robot>/<family>.yaml  # checkpoint (pinned snapshot), dtype, adapter options, max_horizon
├── policies/                      # adapters per family, registered in policies/registry.py
├── policy_server_relay.py         # runs on the Policy Server machine: 127.0.0.1:<port> -> remote model
└── pbs/
    ├── env.sh                     # NSCC paths, venvs, caches, per-embodiment port ranges
    ├── setup.sh                   # login node, once: checkpoints, pi05 venv, Tailscale binaries, check
    ├── interactive.sh             # run models inside an interactive GPU job (qsub -I)
    ├── <family>.pbs               # batch job: one model
    ├── cohost.pbs                 # batch job: several families, one embodiment, one GPU per family
    ├── serve.sh                   # start one family/robot/port from that family's venv
    ├── tailscale.sh               # join the tailnet (TUNNEL=tailscale)
    └── tunnel_cmd.sh              # print an SSH tunnel command for a running job
```

## NSCC submission settings

Taken from `robocolosseum-training/nscc_pbs`, scaled down from 8 GPUs:

| Setting | Value |
|---|---|
| Project | `-P 11010001` |
| Queue | `-q normal`, a routing queue: 1-7 GPUs go to `gdev` (at most 2 h, at most 2 running jobs per user) or `g1` (2-24 h, shared nodes) |
| Resources, `cohost.pbs` | `select=1:ncpus=60:ngpus=5:mem=900gb`: one GPU per model, with 12 CPUs and 180 GB per GPU as in training |
| Resources, `<family>.pbs` | `select=1:ncpus=24:ngpus=1:mem=275gb` (1/8 of a 192-core, 2203 GB node) |
| Walltime | `cohost.pbs` `08:00:00`, `<family>.pbs` `02:00:00`; override with `-l walltime=...` |
| PBS output | `-j oe`, `-o /data/projects/51001013/tansang/logs/pbs/` |

Host RAM peaked at 96 GB with four families on Hopper. Models get GPUs in `MODELS` order, and the mapping
is printed in the PBS output. With fewer GPUs than models (override `-l select=...`), models share GPUs in turn.

## Ports

Each server binds `127.0.0.1` on the GPU node, starting at 9101 (so101), 9111 (yam) or 9121 (franka).
`g1` jobs share nodes, so two embodiments can land on the same node. Set `BASE_PORT` (`cohost.pbs`,
`interactive.sh`) or `PORT` (`<family>.pbs`) to override this. `/act` has no authentication, and
other users' jobs on a shared node can reach loopback ports.

## Embodiments

`--robot` (`EMBODIMENT` in the PBS scripts) selects the request layout:

| `--robot` | Image fields | `state` / each action row | `control_hz` |
|---|---|---|---|
| `so101` | `external_cam`, `wrist_cam` | 6: 5 joints (deg), gripper [0, 100] | 30 (g05: 15) |
| `yam` | `top_cam`, `left_cam`, `right_cam` | 14: left 6 joints, left gripper, right 6 joints, right gripper; grippers [0, 1], 1 = open | 30 |
| `franka` (DROID) | `external_cam`, `wrist_cam` | 8: 7 joints (rad), gripper [0, 1] | 15 |

Every action row is an absolute joint target in the state layout. All so101
checkpoints below were trained in LeRobot's **legacy** SO100 frame, which the
Policy Server converts to by default (`joint_frame: legacy`), so leave it unset.
g1 is not supported: the Policy Server has no g1 contract.

## Models

All were verified on an H200 with random images and a zero state. Each returned
finite chunks of the right shape, rejected a bad request with 500, and answered
`/healthz`. Latency is a steady `/act` round trip at 480×640.

| Robot | Family | Checkpoint @ revision | Chunk (`max_horizon`) | `/act` | GPU mem | Policy Server `adapter` |
|---|---|---|---|---|---|---|
| so101 | molmoact2 | `allenai/MolmoAct2-SO100_101` @ `152569fe57914d97be91055800035f54e250d009` | 30 | 0.16 s | 12.8 GB | `molmoact2_so101` |
| so101 | pi05 | `hqfang/pi05-so100_101` @ `d3204e03ae84d232e2493a933fd460515904eb58` | 50 | 0.19 s | 17.2 GB | `pi05_so101` |
| so101 | groot | `hqfang/groot-so100_101` @ `f5e17af8948a647e4c23186a27927bf73d6d784e` | 40 | 0.14 s | 14.1 GB | `groot_so101` |
| so101 | g05 | `OpenGalaxea/G05` (`g05-so101/`) @ `e312be81e90c56a55bcb26b57429bd39a335b449` | 30 | 1.5 s | 17.2 GB | `molmoact2_so101` ¹ |
| so101 | lingbot | `RoboColosseum/LingBot-MolmoAct2-So101` @ `af0bbb684bf0579d1c4ba19cfd11b8ebd0de1e18` ⁴ | 50 | 0.24 s | 13.2 GB | `lingbot_so101` |
| yam | molmoact2 | `allenai/MolmoAct2-BimanualYAM` @ `8dcbed66f2380e4393189c303ea72488eb9e63c2` | 30 | 0.18 s | 12.4 GB | `molmoact2_yam` |
| yam | pi05 | `NUSMAGIC/pi05-MolmoAct2-YAM` @ `2f28d00ac28c543626f5a4f579a3da09bee4a4ed` | 30 | 0.18 s | 10.2 GB | `pi05_yam` |
| yam | groot | `NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM` @ `5ba9c403bdb962b88c39f843da536eb8370b61b2` | 16 | 0.15 s | 14.1 GB | `groot_yam` |
| yam | lingbot | `helen9975/lingbot-vla-v2-6b-molmoact2-yam` @ `af7529f2ea27ff6e77911a98f86c40cfb75d229e` ² | 30 | 0.25 s | 13.2 GB | `lingbot_yam` |
| yam | g05 | `RoboColosseum/G05-MolmoAct2-YAM` @ `26c91270ce33325a29990b3ff7ea7a4b47bd4ec1` ³ | 32 | 1.4 s | 18.1 GB | `g05_yam` |
| franka | molmoact2 | `allenai/MolmoAct2-DROID` @ `d8c1abd8a27d8e859455bbe514df2bcc617db0fb` | 15 | 0.16 s | 12.4 GB | `molmoact2` |

¹ The Policy Server's `g05_so101` speaks WebSocket. Any so101 HTTP adapter name
sends the same `/act` request.
² Weights are in `checkpoints/global_step_40000/hf_ckpt` (`subfolder` in the
Policy Server entry). It uses `torch.compile`: ready after ~3 min, 4.5 min cold.
Set `use_compile: false` for 0.72 s per call and a faster start.

Notes:
- **g05 so101 latency.** It is slow because its saved config generates
  chain-of-thought text and decodes actions twice. The Policy Server's own G05
  worker runs the same path.
- **g05 so101 rate: 15 Hz.** G05 conditions its action decoding on a `frequency`
  input, and this checkpoint's SO-100 deployment code runs at 15 Hz, so
  `configs/so101/g05.yaml` sets `frequency: 15`. Register it with `control_hz: 15`;
  the other so101 models use 30.
- **g05 assets.** They are in `/data/projects/51001013/tansang/cache/G05-e312be81`, downloaded at the
  pinned revision. `projects/GalaxeaVLA/checkpoints` belongs to training and is left untouched.
- **franka.** `setup.sh` does not download `allenai/MolmoAct2-DROID`; fetch it at the revision above before serving franka.
- **`num_steps`.** It is used by molmoact2 and pi05. groot (4 steps), lingbot and
  g05 (10 steps) keep their checkpoint's setting.

⁴ Weights are in `checkpoint-100000` (`subfolder` in the Policy Server entry); the repo also has `checkpoint-90000`, and neither has been evaluated on a robot. The same weights are public as `RoboColosseum/so101-lingbot-vla-v2` @ cdcbea5b. Uses `torch.compile` like lingbot yam (`use_compile: false`: 0.70 s per call). `control_hz: 30` is assumed.

³ The weights are identical at 685add3f, 3d0d9bb1 and 26c91270; use whichever revision the Router registered (the Policy Server example uses 685add3f). Like g05 so101, it is slow (~1.4 s per 32-step chunk) because the saved config decodes actions twice.

### Environments (serve.sh)

No containers; each family runs from a uv venv in `/scratch/users/nus/duanj1/tansang/venvs/`:

| Family | Venv | Compared with Hopper |
|---|---|---|
| molmoact2 | `lerobot-v0.6.1` (shared with training) | transformers 5.5.4 instead of 5.16 |
| pi05 | `/data/projects/51001013/tansang/venvs/colosseum-pi05` (`setup.sh pi05`: LeRobot b6ec006 + the checkpoint's `lerobot.patch`, source in `/data/projects/51001013/tansang/deps/lerobot-pi05-so101`) | same LeRobot commit and patch |
| groot | `lerobot-v0.6.1` (LeRobot v0.6.1 in `projects/lerobot`) | same LeRobot version |
| lingbot | `lingbot-vla-v2` (`projects/lingbot-vla-v2` @ be969b8) | Hopper used 4144e9c; inference code was recorded as identical at be969b8 |
| g05 | `g05` (`projects/GalaxeaVLA` @ 89f2322) | same commit |

## Co-hosting measurements

On NSCC, `cohost.pbs` gives each model its own GPU, so the numbers below apply only if you share a GPU.
Each process holds its own copy of its model, so GPU memory adds up. A process
called alone runs at its normal speed. Calls that overlap share the GPU and each
slows down.

Measured on one Hopper H200, with four families per job, all loaded in parallel. Each
client sent 480×640 requests back to back. "All at once" means all four clients
ran together, the worst case. Normally the Router drives one model at a time per rig.

| Job | GPU memory | Ready | molmoact2 | pi05 | groot | 4th family |
|---|---|---|---|---|---|---|
| so101, called alone | 60.8 GB | 155 s | 159 ms | 203 ms | 147 ms | g05 1531 ms |
| so101, all at once | | | 446 ms | 369 ms | 417 ms | g05 1690 ms |
| yam, called alone | 49.5 GB | 155 s | 167 ms | 184 ms | 138 ms | lingbot 235 ms |
| yam, all at once | | | 636 ms | 447 ms | 400 ms | lingbot 935 ms |

Host RAM peaked at 96 GB (so101) and 74 GB (yam) of the job's 225 GB.

## Connecting the Policy Server

The server binds 127.0.0.1. **Colosseum Policy Server only accepts loopback model endpoints**
(`http://127.0.0.1:<port>`; see its `backends/base.py`). So each model is reached through a tunnel that
the GPU job opens outwards, plus a small relay on the Policy Server machine that turns it back into a
local port.

### Tailscale (recommended)

One-time setup:
1. Run `bash nscc_pbs_remote_inference/pbs/setup.sh tailscale` on the login node.
2. Put a reusable, ephemeral Tailscale auth key in `nscc_pbs_remote_inference/.env` as
   `TS_AUTHKEY=tskey-auth-...`. The file is git-ignored.
3. Install Tailscale on the Policy Server machine and log in to the same tailnet. For several operators'
   laptops, use a dedicated tailnet: tag the GPU key `tag:colosseum-gpu`, give laptops a
   `tag:policy-server` key, and allow only `tag:policy-server -> tag:colosseum-gpu:9101-9130`.

Each run:
1. Start the models with `interactive.sh ... tailscale` or a batch job with `TUNNEL=tailscale` (see README).
2. On the Policy Server machine, check that `tailscale status` lists `colosseum-<robot>`.
   `curl http://colosseum-so101:9101/healthz` should return `{"status": "ok"}`.
3. Run one relay per model (Python 3 standard library only):
   ```bash
   python3 policy_server_relay.py --remote http://colosseum-so101:9101 --port 9201
   python3 policy_server_relay.py --remote http://colosseum-yam:9111 --port 9211
   ```
4. In the Policy Server config, set each model's `endpoint` to its relay (`http://127.0.0.1:9201`, ...).

Tailscale needs outbound access from the GPU node. The training smoke test confirmed HTTPS access from
NSCC GPU nodes; Tailscale itself has not been run there yet.

### Other options

None of these has been tested on NSCC.

| Option | Notes |
|---|---|
| SSH tunnel | `bash nscc_pbs_remote_inference/pbs/tunnel_cmd.sh <jobid>` prints the command; needs SSH access from the Policy Server machine to NSCC. |
| `TUNNEL=ngrok` | Put the binary at `/scratch/users/nus/duanj1/tansang/tools/ngrok/ngrok`. Public `https://` URL with mandatory basic auth (`NGROK_BASIC_AUTH=user:password,...` and `NGROK_AUTHTOKEN` in `.env`). The relay needs `COLOSSEUM_TUNNEL_AUTH=user:password`. |
| `TUNNEL=cloudflare` | Put the binary at `/scratch/users/nus/duanj1/tansang/tools/cloudflared/cloudflared`. Needs TCP/UDP 7844, which Hopper blocked. |

Each request carries raw images as base64: ~2.5 MB for so101 and ~3.7 MB for yam, about one request per
second at 30 Hz. That is ~9-13 GB per hour of evaluation and ~20-30 Mbit/s of upload from the Policy Server
machine.

## HTTP contract

- `GET /healthz` returns `200 {"status": "ok"}`.
- `POST /act` takes a JSON body with:
  - the robot's image fields (uint8 RGB `(H, W, 3)`)
  - `state` (float32)
  - `instruction` (str)
  - `num_steps` (1–10)

  Other fields are ignored. Arrays arrive as
  `{"__numpy__": <base64 raw bytes>, "dtype": "|u1", "shape": [...]}`.
- On success it returns `200 {"actions": [[...]]}` with shape `(N, action_dim)`,
  `1 <= N <= max_horizon`, all values finite. On failure it returns
  `500 {"error": "..."}` and prints the traceback to stderr.

## Policy Server entry (example)

Fill `url`, `revision`, `max_horizon` and `adapter` from the Models table.
`action_dim` is 6 (so101), 14 (yam) or 8 (franka). `endpoint` is the model's local
relay (or SSH-tunnel) port on the Policy Server machine. Leave `joint_frame` unset for
so101.

```yaml
- name: <as registered on the Robo Colosseum Router>
  url: https://huggingface.co/hqfang/pi05-so100_101
  revision: "d3204e03ae84d232e2493a933fd460515904eb58"
  action_space: joint_position
  action_dim: 6
  control_hz: 30
  max_horizon: 50
  endpoint: http://127.0.0.1:9201
  backend_options: {robot_type: so101, adapter: pi05_so101, num_steps: 10}
```

## Adding a model

1. Download the pinned revision on the login node:
   `HF_HOME=/data/projects/51001013/tansang/cache/hf /scratch/users/nus/duanj1/tansang/venvs/hftools/bin/hf download <org>/<repo> --revision <sha>`.
2. Write `configs/<robot>/<family>.yaml`. Set `checkpoint` to the snapshot path, and
   add the family's adapter options; copy them from an existing config of that family.
3. Start it (`interactive.sh <robot> <family>` or `qsub -v EMBODIMENT=<robot> nscc_pbs_remote_inference/pbs/<family>.pbs`)
   and test it with random requests before registering it.
