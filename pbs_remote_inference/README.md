# pbs_remote_inference

Run a large VLA model on an HPC PBS GPU node while a robot is controlled locally.

## What runs where

**On the HPC (server side):**
- `launcher_server/` — the launcher API. You start it once. It takes model
  requests and submits GPU jobs with `qsub`.
- `worker/` — the GPU worker. The launcher starts it inside a PBS job. It loads
  the model once and connects out to the local machine to do inference.

**On the collaborator's machine (local side):**
- `local/` — the robot program (`run_robot.py`). It starts a local WebSocket
  server, asks the launcher for a model, receives observations from the robot,
  sends them to the GPU worker, and runs the returned actions.

**Shared code (used on both sides):**
- `models.py` — model name → policy / PBS script / config.
- `protocol.py` — WebSocket message format + JPEG image encoding.

```
LOCAL: run_robot.py ──POST /jobs──▶ HPC: launcher ──qsub──▶ HPC: GPU worker
       (WebSocket server)                                        │
              ▲─────────────── observations / actions (WSS) ─────┘
```

The launcher only starts jobs. Observations and actions go straight between the
local machine and the GPU worker, not through the launcher.

## Install

```bash
pip install -r pbs_remote_inference/requirements.txt
```

Model deps (torch, transformers, ...) come from each policy's own environment.

## Test it locally (no PBS, no GPU, no ngrok)

Run both sides on one machine with a dummy model.

```bash
# Terminal 1 — local side (server + dummy robot)
python pbs_remote_inference/local/run_robot.py \
    --model dummy --offline-session-id session-local \
    --callback-token test-token --port 8000 --max-steps 3

# Terminal 2 — worker side (dummy model)
python pbs_remote_inference/worker/run_worker.py \
    --model dummy --server ws://127.0.0.1:8000/worker \
    --token test-token --session-id session-local
```

Expect the local side to log `Worker connected and ready` then `Action received`.

## Real run

### On the HPC: start the launcher

```bash
export LAUNCHER_API_TOKEN="a-long-random-key"
python -m uvicorn pbs_remote_inference.launcher_server.app:app --host 127.0.0.1 --port 8000
ngrok http 8000            # -> https://MY-LAUNCHER.ngrok-free.app
```

(Or submit it as a CPU job: `qsub pbs/launcher.pbs`.) The GPU worker jobs are
started automatically by the launcher — you do not run `qsub` per request.

### On the local machine: run the robot

Point your own ngrok at your local port 8000, then:

```bash
python pbs_remote_inference/local/run_robot.py \
    --model molmoact2-droid \
    --launcher-url https://MY-LAUNCHER.ngrok-free.app \
    --callback-url wss://THEIR-ROBOT.ngrok-free.app/worker \
    --launcher-token "$LAUNCHER_API_TOKEN" --port 8000
```

It requests the model, waits for the worker to connect back, then streams
observations and runs the returned action chunks. Ctrl-C cancels the job
(unless `--keep-job`). Version 1 ships `DummyRobot`; for a real robot, implement
`RobotInterface` and use it in `run_robot.py`.

## Launcher API

All endpoints need `Authorization: Bearer $LAUNCHER_API_TOKEN`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness + `qstat` availability |
| `GET /models` | list model names |
| `POST /jobs` | validate + `qsub`, returns `session_id`/`job_id` |
| `GET /jobs/{session_id}` | session status (refreshed via `qstat`) |
| `DELETE /jobs/{session_id}` | `qdel` the job, mark `cancelled` |

`POST /jobs` body: `{"model", "callback_url", "callback_token"}` →
`{"session_id", "job_id", "model", "status"}`.
Statuses: `queued`, `running`, `finished`, `failed`, `cancelled`.

## Protocol (local ⇄ worker)

JSON frames validated by pydantic: `worker_ready` → `worker_ack`, then
`observation` ⇄ `action` (or `error`). Cameras are base64 JPEG under generic
slot names (`external`, `wrist`, ...); actions are a `(horizon, action_dim)`
chunk. See [protocol.py](protocol.py).

## Safety & security (all on the local side)

- Before running a chunk, the local side checks `request_id` match, latency ≤
  timeout, 2-D shape, no NaN/Inf, and optional dim/horizon/value limits. On
  timeout or disconnect the robot enters `safe_stop()`; old actions are never
  replayed. See [local/safety.py](local/safety.py).
- Two tokens: launcher API key (for the launcher) and per-session callback token
  (for the local ⇄ worker link) — never the same, never logged in full.
- Callback URL allowlist: only `wss://` on approved ngrok domains; IPs,
  localhost and private ranges are rejected. Override with
  `LAUNCHER_ALLOWED_CALLBACK_DOMAINS`.
- Model names map to fixed PBS scripts (no user paths); `qdel` only ever targets
  a known session's job id; subprocess calls use list form (no shell).
