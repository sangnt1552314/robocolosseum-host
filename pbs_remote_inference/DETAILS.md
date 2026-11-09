# pbs_remote_inference: details

Reference for [README.md](README.md): models, environments, measurements, tunnel options, HTTP contract.

## Layout

```
pbs_remote_inference/
├── colosseum_server.py            # HTTP server (stdlib + numpy): GET /healthz, POST /act
├── configs/<robot>/<family>.yaml  # checkpoint (pinned snapshot), dtype, adapter options, max_horizon
├── policies/                      # adapters per family, registered in policies/registry.py
├── policy_server_relay.py         # runs on the Policy Server machine: 127.0.0.1:<port> -> remote model
└── pbs/
    ├── interactive.sh             # run models inside an interactive GPU job (qsub -I)
    ├── <family>.pbs               # batch job: one model
    ├── cohost.pbs                 # batch job: several families, one embodiment, one GPU
    ├── serve.sh                   # start one family/robot/port in that family's container + venv
    ├── tailscale.sh               # join the tailnet (TUNNEL=tailscale)
    └── tunnel_cmd.sh              # print an SSH tunnel command for a running job
```

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
- **g05 so101 weights.** It reads the existing download in
  `/scratch/e1583535/projects/GalaxeaVLA/checkpoints`.
- **`num_steps`.** It is used by molmoact2 and pi05. groot (4 steps), lingbot and
  g05 (10 steps) keep their checkpoint's setting.

⁴ Weights are in `checkpoint-100000` (`subfolder` in the Policy Server entry); the repo also has `checkpoint-90000`, and neither has been evaluated on a robot. The same weights are public as `RoboColosseum/so101-lingbot-vla-v2` @ cdcbea5b. Uses `torch.compile` like lingbot yam (`use_compile: false`: 0.70 s per call). `control_hz: 30` is assumed.

³ The weights are identical at 685add3f, 3d0d9bb1 and 26c91270; use whichever revision the Router registered (the Policy Server example uses 685add3f). Like g05 so101, it is slow (~1.4 s per 32-step chunk) because the saved config decodes actions twice.

### Environments (serve.sh)

| Family | Container (`/app1/common/singularity-img/hopper/pytorch/`) | Venv (`/scratch/e1583535/virtualenvs/`) |
|---|---|---|
| molmoact2 | `pytorch_2.6.0_cuda_12.8.sif` | `robocolosseum` |
| pi05 | `pytorch_2.7.0_cuda_12.8.sif` | `colosseum-pi05` (LeRobot b6ec006 + the checkpoint's `lerobot.patch`, `projects/colosseum-deps/lerobot-pi05-so101`) |
| groot | `pytorch_2.7.0_cuda_12.8.sif` | `lerobot-cu128` (editable LeRobot v0.6.1 in `projects/lerobot`, shared with training) |
| lingbot | `pytorch_2.6.0_cuda_12.8.sif` | `lingbot-vla-v2` (`projects/lingbot-vla-v2` @ 4144e9c) |
| g05 | `pytorch_2.3.0_cuda_12.4_ngc_24.04.sif` | `g05` (`projects/colosseum-deps/GalaxeaVLA` @ 89f2322) |

## Co-hosting measurements

Each process holds its own copy of its model, so GPU memory adds up. A process
called alone runs at its normal speed. Calls that overlap share the GPU and each
slows down.

Measured on one H200, with four families per job, all loaded in parallel. Each
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

The server binds 127.0.0.1, and Hopper GPU nodes accept no inbound connections. Also,
**Colosseum Policy Server only accepts loopback model endpoints**
(`http://127.0.0.1:<port>`; see its `backends/base.py`). So each model is reached
through a tunnel that the GPU job opens outwards, plus a small relay on the Policy
Server machine that turns it back into a local port.

### Tailscale (recommended; tested end to end)

One-time setup:
1. Put a reusable, ephemeral Tailscale auth key in `pbs_remote_inference/.env` as
   `TS_AUTHKEY=tskey-auth-...`. The file is git-ignored.
2. Install Tailscale on the Policy Server machine and log in to the same tailnet. For
   several operators' laptops, use a dedicated tailnet: tag the GPU key
   `tag:colosseum-gpu`, give laptops a `tag:policy-server` key, and allow only
   `tag:policy-server -> tag:colosseum-gpu:9101-9110` in the access controls.

Each run:
1. Start the models with `interactive.sh ... tailscale` or a batch job with `TUNNEL=tailscale` (see README).
2. On the Policy Server machine, check that `tailscale status` lists `colosseum-<robot>`.
   `curl http://colosseum-so101:9101/healthz` should return `{"status": "ok"}`.
3. Run one relay per model (Python 3 standard library only). Copy the script from this
   repo:
   ```bash
   python3 policy_server_relay.py --remote http://colosseum-so101:9101 --port 9201
   python3 policy_server_relay.py --remote http://colosseum-so101:9102 --port 9202
   ```
4. In the Policy Server config, set each model's `endpoint` to its relay
   (`http://127.0.0.1:9201`, ...); see the entry example below.

Tested from a MacBook through the Policy Server's own so101 inference path:
- MolmoAct2 returned 30×6 and GR00T 40×6.
- Each call took ~1.0 s on the server. About 0.15 s is inference; the rest is the
  2.5 MB of images arriving via Tailscale's relay, because Hopper blocks UDP, so
  there is no direct path.

### Other options

| Option | Status | Notes |
|---|---|---|
| SSH tunnel | works | Only if the Policy Server machine can SSH into Hopper. `bash pbs_remote_inference/pbs/tunnel_cmd.sh <jobid>` prints `ssh -N -J <you>@<login> -L 9201:127.0.0.1:9101 ... <you>@<gpu-node-ip>`; endpoint `http://127.0.0.1:9201`. GPU nodes allow port forwarding but kill remote commands. Add `-o GSSAPIAuthentication=no` to avoid a Kerberos hang. |
| `TUNNEL=ngrok` | works | Public `https://` URL with mandatory basic auth (`NGROK_BASIC_AUTH=user:password,...` in `.env`, one per operator). The Policy Server machine runs the relay with `COLOSSEUM_TUNNEL_AUTH=user:password`. Free plan: one model at a time, 1 GB/month (~400 so101 requests), so real evaluations need a paid plan. |
| `TUNNEL=cloudflare` | blocked on Hopper | `cloudflared` needs TCP/UDP 7844, which GPU nodes block. |

Measured from Hopper GPU nodes: outbound TCP 443 works, while UDP and TCP 7844 are
blocked. Each request carries raw images as base64: ~2.5 MB for so101 and ~3.7 MB for
yam, about one request per second at 30 Hz. That is ~9-13 GB per hour of evaluation
and ~20-30 Mbit/s of upload from the Policy Server machine.

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
   `HF_HOME=/scratch/e1583535/cache hf download <org>/<repo> --revision <sha>`.
2. Write `configs/<robot>/<family>.yaml`. Set `checkpoint` to the snapshot path, and
   add the family's adapter options; copy them from an existing config of that family.
3. Start it (`interactive.sh <robot> <family>` or `qsub -v EMBODIMENT=<robot> pbs/<family>.pbs`)
   and test it with random requests before registering it.
