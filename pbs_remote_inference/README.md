# pbs_remote_inference

Run a large VLA model on an HPC PBS GPU node while the robot is controlled
locally. A launcher API on the HPC submits GPU jobs; each job loads the model
once and connects out (WebSocket) to the local robot to exchange observations
and action chunks. Observations/actions never pass through the launcher.

```
LOCAL: run_robot.py ──POST /jobs──▶ HPC: launcher ──qsub──▶ HPC: GPU worker
       (WebSocket server)                                        │
              ▲─────────────── observations / actions (WSS) ─────┘
```

## Directory structure

```
pbs_remote_inference/
├── models.py          # model name → policy / PBS script / config
├── protocol.py        # WebSocket message format + JPEG encoding
├── requirements.txt
├── configs/           # worker YAML, per embodiment/task: configs/<embodiment>/<task>/<policy>.yaml
├── launcher_server/   # HPC control-plane API (submits/cancels PBS jobs)
├── worker/            # HPC GPU worker (loads model, dials back to local)
├── local/             # robot machine: run_robot.py + embodiments + safety
└── pbs/               # PBS job scripts (one per model + launcher.pbs)
```

The `.pbs` scripts pick their config with `EMBODIMENT` (default `so101`) and
`TASK` (default `stack_white_bowls_100episodes`); override via
`qsub -v EMBODIMENT=...,TASK=...` to run another embodiment/task later.

## Installation

```bash
pip install -r pbs_remote_inference/requirements.txt
```

Model dependencies (torch, transformers, ...) come from each policy's own
environment, not this file.

## How to run

Models: `molmoact2`, `pi05`, `groot`, `g05`, `lingbot`
(or `dummy` for a no-GPU test).

### HPC — start the launcher

```bash
export LAUNCHER_API_TOKEN="a-long-random-key"
python -m uvicorn pbs_remote_inference.launcher_server.app:app --host 127.0.0.1 --port 8000
ngrok http 8000        # → https://MY-LAUNCHER.ngrok-free.app
```

The launcher submits the GPU worker jobs itself — you never run `qsub` per
request.

### Local — run the robot

`local/run_robot.py` starts the local server, requests a model, and runs the
returned actions on the robot.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--model` | yes | Model name (see list above). |
| `--embodiment` | no | Robot to drive: `so101` (default) or `dummy`. |
| `--launcher-url` | yes | Launcher API URL (`https://...`). |
| `--callback-url` | yes | Public `wss://` URL of this machine's worker endpoint. |
| `--launcher-token` | yes | Launcher API key (or `LAUNCHER_API_TOKEN`). |
| `--follower-port` | so101 | SO101 serial port (or `FOLLOWER_PORT`). |
| `--follower-id` | no | SO101 id (or `FOLLOWER_ID`, default `home_follower`). |
| `--wrist-camera` / `--external-camera` | no | OpenCV camera indices (default `0` / `1`). |
| `--execute-steps` | no | Actions to run per chunk (default `4`). |
| `--step-delay` | no | Seconds between actions (default `0.1`). |
| `--port` | no | Local server port (default `8000`). |
| `--max-steps` | no | Stop after N chunks (default: unlimited). |
| `--keep-job` | no | Do not cancel the PBS job on exit. |

```bash
python pbs_remote_inference/local/run_robot.py \
    --model molmoact2 --embodiment so101 --follower-port /dev/ttyACM0 \
    --launcher-url https://MY-LAUNCHER.ngrok-free.app \
    --callback-url wss://THEIR-ROBOT.ngrok-free.app/worker \
    --launcher-token "$LAUNCHER_API_TOKEN"
```

Add a new robot (franka, bimanual_yam, g1, ...) by implementing `RobotInterface`
and registering a factory in `local/embodiments.py`.

### Local test (no PBS, no GPU, no ngrok)

Run both sides on one machine with the dummy model and robot:

```bash
# Terminal 1 — local server + dummy robot
python pbs_remote_inference/local/run_robot.py \
    --model dummy --embodiment dummy --offline-session-id session-local \
    --callback-token test-token --port 8000 --max-steps 3

# Terminal 2 — dummy worker
python pbs_remote_inference/worker/run_worker.py \
    --model dummy --server ws://127.0.0.1:8000/worker \
    --token test-token --session-id session-local
```

### Worker configs

The GPU worker loads a YAML at
`configs/<embodiment>/<task>/<policy>.yaml` (e.g.
`configs/so101/stack_white_bowls_100episodes/molmoact2.yaml`). Only two keys are
required; everything else falls back to defaults:

```yaml
policy:
  name: molmoact2          # adapter registry key
  checkpoint: /path/to/local/checkpoint
```

Optional: `policy.dtype`/`device`, adapter extras (e.g. `action_dim`,
`g05_repo`), `camera_mapping` (`external`/`wrist`), and `runtime`
(`action_spaces`, `control_hz`, `max_horizon`). To support a new embodiment or
task, add `configs/<embodiment>/<task>/<policy>.yaml` and submit with
`qsub -v EMBODIMENT=...,TASK=...` — no script changes needed.

## API docs

All endpoints require `Authorization: Bearer $LAUNCHER_API_TOKEN`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness + `qstat` availability |
| `GET /models` | list model names |
| `POST /jobs` | validate + `qsub`, returns `session_id` / `job_id` |
| `GET /jobs/{session_id}` | session status (refreshed via `qstat`) |
| `DELETE /jobs/{session_id}` | `qdel` the job, mark `cancelled` |

`POST /jobs` body: `{"model", "callback_url", "callback_token"}` →
`{"session_id", "job_id", "model", "status"}`.
Statuses: `queued`, `running`, `finished`, `failed`, `cancelled`.

Security: callback URLs must be `wss://` on an approved ngrok domain (override
with `LAUNCHER_ALLOWED_CALLBACK_DOMAINS`); model names map to fixed PBS scripts
(no user paths); subprocess calls use list form (no shell). Every action chunk
is validated locally (request-id match, latency, shape, NaN/Inf, optional
limits) before execution, and the robot enters `safe_stop()` on timeout or
disconnect. See `local/safety.py` and `protocol.py`.
