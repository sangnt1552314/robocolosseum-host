# pbs_remote_inference — launcher-orchestrated robot ⇄ HPC inference

A collaborator requests a VLA model over a small HTTP **launcher API**. The
launcher submits a **PBS GPU worker** with `qsub`. The worker loads the existing
policy adapter, then connects **outward** (WSS via the user's ngrok) to the
user's own local WebSocket server and exchanges observations/action chunks
directly. **The launcher is never in the observation/action data path.**

This workflow is independent of `slurm_job_launcher_server/` and of any external
policy-server SDK. It only reuses the policy adapters in
[`src/robocolosseum/policies`](../src/robocolosseum/policies).

```
Robot user ──POST /jobs──▶ Launcher API ──qsub──▶ PBS GPU worker
                                                        │
                                                        │ outbound WSS
                                                        ▼
                                          Robot user's ngrok URL
                                                        │
                                                        ▼
                                                  Local robot
```

Two planes:

- **Control plane** (HTTP): request a model, submit/query/cancel the PBS job.
- **Data plane** (WSS): observations → inference → action chunks, directly
  between the local robot server and the GPU worker.

## Layout

```
pbs_remote_inference/
├── models.py                     # public name → (policy, PBS script, config)
├── protocol.py                   # shared JSON message schemas + JPEG codec
├── launcher_server/              # control plane (FastAPI)
│   ├── app.py                    # /health /models /jobs ...
│   ├── job_manager.py            # qsub/qstat/qdel + in-memory sessions
│   ├── model_registry.py         # strict name → script registry
│   ├── callback_validation.py    # SSRF allowlist for callback_url
│   ├── auth.py                   # Bearer LAUNCHER_API_TOKEN
│   ├── errors.py / schemas.py
├── local/                        # the robot user's machine
│   ├── run_robot.py              # orchestrator CLI
│   ├── robot_server.py           # WebSocket callback server + control loop
│   ├── launcher_client.py        # HTTP client for the launcher API
│   ├── robot_interface.py        # RobotInterface ABC + DummyRobot
│   └── safety.py                 # action validation + SafetyLimits
├── worker/                       # the PBS GPU job
│   ├── run_worker.py
│   ├── inference_worker.py       # load model once, connect out, infer
│   └── observation.py            # bridge message → adapter observation
├── pbs/                          # launcher.pbs + one script per model
└── requirements.txt
```

## Install

```bash
pip install -r pbs_remote_inference/requirements.txt
```

Model dependencies (torch, transformers, lerobot, gr00t, ...) come from each
policy's own environment and are not listed here.

## Launcher API

All endpoints require `Authorization: Bearer $LAUNCHER_API_TOKEN`.

| Method & path | Purpose |
|---------------|---------|
| `GET /health` | liveness + whether `qstat` is available |
| `GET /models` | list registered public model names |
| `POST /jobs` | validate model + callback URL, `qsub` the worker, return `session_id`/`job_id` |
| `GET /jobs/{session_id}` | session info; refreshes status via `qstat` |
| `DELETE /jobs/{session_id}` | `qdel` the session's job, mark `cancelled` |

`POST /jobs` body:

```json
{
  "model": "molmoact2-droid",
  "callback_url": "wss://robot-user.ngrok-free.app/worker",
  "callback_token": "temporary-session-token"
}
```

Response (`202`):

```json
{ "session_id": "session-abc123", "job_id": "1234567", "model": "molmoact2-droid", "status": "queued" }
```

Statuses: `queued`, `running`, `finished`, `failed`, `cancelled` (derived from
PBS), plus `submitting`/`failed` around submission.

### Security

- **Two credentials**: the launcher API key (`LAUNCHER_API_TOKEN`, control
  plane) and a per-session **callback token** (data plane) — never the same,
  never logged in full.
- **Model registry** is strict: users send only a public model name; the
  launcher picks the pre-approved PBS script. No user-supplied paths/args.
- **Callback URL allowlist** ([callback_validation.py](callback_validation.py)):
  only `wss://` on approved tunnel domains (`*.ngrok-free.app`,
  `*.ngrok-free.dev`, `*.ngrok.app`, `*.ngrok.io`, `*.ngrok.dev`). IP literals,
  `localhost`, and private ranges are rejected so the worker can't be used as a
  proxy into internal infrastructure. Override the allowlist with
  `LAUNCHER_ALLOWED_CALLBACK_DOMAINS` (comma-separated).
- **`qdel` safety**: cancellation only ever targets the job id recorded for a
  known session; users cannot pass arbitrary PBS job ids.
- **Subprocess** calls use list form (no shell); `qsub -v` values are validated.

## WebSocket protocol (data plane)

JSON text frames, validated with pydantic (unknown fields rejected):

| Direction | `type` | Key fields |
|-----------|--------|-----------|
| worker → local | `worker_ready` | `session_id`, `model`, `worker_id`, `action_horizon` |
| local → worker | `worker_ack` | `session_id`, `accepted`, `message` |
| local → worker | `observation` | `request_id`, `timestamp`, `instruction`, `robot_state`, `cameras`, `metadata` |
| worker → local | `action` | `request_id`, `model`, `actions` (chunk), `inference_time_ms` |
| worker → local | `error` | `request_id`, `message` |

- **Images**: `cameras` maps a generic slot (e.g. `external`, `wrist`, `left`,
  `right`) to a base64 **JPEG**. The codec is isolated in
  `encode_jpeg`/`decode_jpeg`, so binary frames can be added later without
  changing the schemas.
- **Action chunks**: `actions` is a `(horizon, action_dim)` list.
- The worker maps generic `external`/`wrist` slots onto each adapter's
  `camera_mapping` attribute names — no model code is modified.

## Safety (local only)

Before any chunk is executed, [local/safety.py](local/safety.py) checks the
`request_id` match (stale rejection), round-trip latency ≤ timeout, 2-D shape,
no `NaN`/`Inf`, and optional `expected_action_dim`/`max_horizon`/value limits.
On timeout, disconnect or any failed check the robot enters `safe_stop()` and
old actions are never replayed.

---

## How I start the launcher (operator)

On a small persistent host or CPU PBS job:

```bash
export LAUNCHER_API_TOKEN="a-long-random-launcher-key"
python -m uvicorn pbs_remote_inference.launcher_server.app:app --host 127.0.0.1 --port 8000
# or: qsub -v LAUNCHER_API_TOKEN pbs_remote_inference/pbs/launcher.pbs
```

Expose it (separate terminal); v1 does not manage ngrok for you:

```bash
ngrok http 8000     # e.g. https://MY-LAUNCHER.ngrok-free.app
```

Quick manual checks:

```bash
curl -H "Authorization: Bearer $LAUNCHER_API_TOKEN" https://MY-LAUNCHER.ngrok-free.app/health
curl -H "Authorization: Bearer $LAUNCHER_API_TOKEN" https://MY-LAUNCHER.ngrok-free.app/models
```

## How a collaborator runs their robot

1. Expose their local worker port with ngrok (separate terminal):

   ```bash
   ngrok http 8000   # gives wss://THEIR-ROBOT.ngrok-free.app  (use the wss:// form + /worker)
   ```

2. Run the orchestrator (starts the local server, requests the model, waits for
   the worker, then streams observations):

   ```bash
   python pbs_remote_inference/local/run_robot.py \
       --model molmoact2-droid \
       --launcher-url https://MY-LAUNCHER.ngrok-free.app \
       --callback-url wss://THEIR-ROBOT.ngrok-free.app/worker \
       --launcher-token "$LAUNCHER_API_TOKEN" \
       --port 8000 --max-steps 3
   ```

   The user needs no Hopper account, SSH, PBS, or `qsub` knowledge. Version 1
   ships `DummyRobot`; wire a real robot by implementing `RobotInterface` and
   constructing it in `run_robot.py` instead of `DummyRobot`.

## Observation / action flow

`run_robot.py` sends observations with incrementing `request_id`s, waits up to
`--inference-timeout` for the matching action chunk, validates it, and executes
it on the robot. On stop/cancel it calls `DELETE /jobs/{session_id}` (unless
`--keep-job`).

## Stopping / cancelling

- Ctrl-C in `run_robot.py` cancels the job (unless `--keep-job`).
- Manually:

  ```bash
  curl -X DELETE -H "Authorization: Bearer $LAUNCHER_API_TOKEN" \
      https://MY-LAUNCHER.ngrok-free.app/jobs/session-abc123
  ```

---

## Manual local test (no PBS, no GPU, no ngrok)

`run_robot.py --offline-session-id ...` skips the launcher/`qsub` and just runs
the local server, so you can exercise the whole **data plane** with the dummy
model on one machine.

Terminal 1 — local robot server (dummy robot), waiting for a known session:

```bash
python pbs_remote_inference/local/run_robot.py \
    --model dummy \
    --offline-session-id session-local \
    --callback-token test-token \
    --port 8000 --max-steps 3
```

Terminal 2 — worker (dummy model) connecting to the local server:

```bash
python pbs_remote_inference/worker/run_worker.py \
    --model dummy \
    --server ws://127.0.0.1:8000/worker \
    --token test-token \
    --session-id session-local
```

Expected: the server logs `Worker connected and ready`, `Observation sent`,
`Action received ...`, and `DummyRobot executing action chunk ...`; the worker
logs `Observation received`, `Inference done`, and the action send.

## Manual launcher test (control plane)

The launcher's `POST /jobs` calls `qsub`, so run it on the cluster (or a host
with PBS). `GET /health` and `GET /models` work anywhere:

```bash
export LAUNCHER_API_TOKEN=launch-key
python -m uvicorn pbs_remote_inference.launcher_server.app:app --host 127.0.0.1 --port 9000 &
curl -H "Authorization: Bearer launch-key" http://127.0.0.1:9000/models
# callback URL validation (should be rejected: not an approved wss domain):
curl -s -X POST -H "Authorization: Bearer launch-key" -H "Content-Type: application/json" \
    -d '{"model":"molmoact2-droid","callback_url":"wss://127.0.0.1/worker","callback_token":"abcdefgh"}' \
    http://127.0.0.1:9000/jobs
```

The full launcher → PBS → worker → local path runs on the cluster with ngrok, as
described above.
