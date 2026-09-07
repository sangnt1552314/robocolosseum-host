# RoboColosseum Job Launcher (temporary control plane)

A small, self-contained HTTP API that **submits, queries, lists and cancels**
the GPU policy jobs on the NUS SoC Slurm cluster. It runs as a lightweight
CPU-only Slurm job (up to 3 days on the `long` partition) and is exposed to the
outside world through an [ngrok](https://ngrok.com) tunnel.

> This is an intentionally simple V1 — no database, no queue, no Docker. It is
> easy to read, easy to delete, and easy to replace later.

## 1. What this server does

```
Aaron / contractor
      |  HTTPS + Bearer token
      v
  ngrok public URL
      |
      v
FastAPI launcher  (this component, CPU-only Slurm job)
      |  sbatch / squeue / sacct / scancel
      v
   SoC Slurm
      |
      v
GPU policy worker  (slurm/molmoact2.sh) --> MolmoAct2-DROID
```

This **control path** is completely separate from the robot **data path**,
which is unchanged and does not involve this launcher at all:

```
Franka -> Aaron's router -> Colosseum SDK -> GPU policy worker -> MolmoAct2
```

The launcher **never** handles camera streams, robot observations, actions,
Colosseum sessions, or policy inference. It only starts and stops Slurm jobs.

## 2. Prerequisites

- NUS SoC Slurm access (submit host + `sbatch`/`squeue`/`sacct`/`scancel`).
- Python 3.12 (the project's `py312` virtualenv at `/home/n/ntasang/py312`).
- A free [ngrok](https://dashboard.ngrok.com) account (for the authtoken).
- The ngrok binary installed in a user-owned path (see below).

Install the launcher's Python dependencies into your environment:

```bash
source /home/n/ntasang/py312/bin/activate
pip install -r job_launcher_server/requirements.txt
```

## 3. Installing ngrok (no root)

ngrok ships as a single static binary, so you can install it into your home
directory without root. On the SoC cluster (Linux x86-64):

```bash
mkdir -p ~/bin
cd /tmp
# Verify the current download URL for your architecture at:
#   https://ngrok.com/download
curl -L -o ngrok.tgz \
  https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok.tgz
mv ngrok ~/bin/ngrok
chmod +x ~/bin/ngrok
```

> Check `uname -m` first. Use the `linux-amd64` build for `x86_64` and the
> `linux-arm64` build for `aarch64`. Do not blindly trust the URL above — the
> canonical, current links are on the ngrok download page.

Confirm it works:

```bash
~/bin/ngrok version
```

The default binary path in [launcher.example.yaml](launcher.example.yaml) is
`/home/n/ntasang/bin/ngrok`. Override it with the `NGROK_BIN` env var or the
`launcher.ngrok_bin` config key if yours lives elsewhere.

## 4. Configuring ngrok authentication

Create a free ngrok account, copy your authtoken from the dashboard, and store
it once in ngrok's local config:

```bash
~/bin/ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

ngrok writes this to its per-user config file (typically
`~/.config/ngrok/ngrok.yml` on Linux). You only need to do this once per
machine/account.

> **Never commit the ngrok authtoken or `ngrok.yml`.** It grants access to your
> ngrok account.

## 5. Launcher API authentication

Every endpoint (including `/health`) requires:

```
Authorization: Bearer <token>
```

The token is read from the `LAUNCHER_API_TOKEN` environment variable. It is
never logged, never returned in an error, and compared in constant time.

Generate a strong token and export it **before** submitting the launcher job:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
export LAUNCHER_API_TOKEN='<paste-the-generated-token>'
```

> Never hard-code, commit, or print this token. Share it with the contractor
> over a secure channel only.

## 6. Configuration

Copy the example config and edit if needed (the real file is git-ignored):

```bash
cp job_launcher_server/launcher.example.yaml job_launcher_server/launcher.yaml
```

```yaml
launcher:
  host: 127.0.0.1
  port: 8000
  max_active_jobs: 10          # hard ceiling across ALL policy jobs
  ngrok_bin: /home/n/ntasang/bin/ngrok

policies:
  molmoact2-droid:
    script: slurm/molmoact2.sh
    slurm_job_name: colosseum-molmo
    max_replicas: 3            # per-model concurrent cap
```

The HTTP caller may only pick one of the registered **model names**. The script
path, Slurm job name and replica cap are fixed in config — the caller can never
supply an arbitrary script, shell command, or Slurm argument.

## 7. Starting the launcher

From the project root, with `LAUNCHER_API_TOKEN` exported:

```bash
sbatch job_launcher_server/launcher.slurm
```

Find the launcher's Slurm job id:

```bash
squeue --me --name=colosseum-launcher
```

## 8. Finding the ngrok public URL

The launcher prints the URL to its Slurm log once ngrok connects:

```bash
grep "LAUNCHER PUBLIC URL" logs/launcher/slurm-<launcher_job_id>.out
```

Example output:

```
==========================================================
LAUNCHER PUBLIC URL: https://example.ngrok-free.app
==========================================================
```

> On the free ngrok plan the URL changes every time the launcher restarts — do
> not assume a fixed URL.

## 9. API reference

All requests require `Authorization: Bearer $LAUNCHER_API_TOKEN`.
Set a shell variable for convenience:

```bash
URL=https://<ngrok-url>
```

### Health

```bash
curl -H "Authorization: Bearer $LAUNCHER_API_TOKEN" "$URL/health"
# {"status":"ok","slurm_available":true}
```

### Submit a job

```bash
curl -X POST \
  -H "Authorization: Bearer $LAUNCHER_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"molmoact2-droid"}' \
  "$URL/jobs"
# 202 {"job_id":"825501","model":"molmoact2-droid","state":"PENDING","request_id":null}
```

Optionally pass a `request_id` for idempotency. Re-submitting the same
non-empty `request_id` returns the existing job instead of creating a duplicate:

```bash
curl -X POST \
  -H "Authorization: Bearer $LAUNCHER_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"molmoact2-droid","request_id":"match-123-policy-a"}' \
  "$URL/jobs"
```

### List launcher-owned jobs

```bash
curl -H "Authorization: Bearer $LAUNCHER_API_TOKEN" "$URL/jobs"
# {"jobs":[{"job_id":"825501","model":"molmoact2-droid","state":"RUNNING"}, ...]}
```

Only jobs this launcher recognises as registered policy jobs are returned.

### Job status

```bash
curl -H "Authorization: Bearer $LAUNCHER_API_TOKEN" "$URL/jobs/825501"
# {"job_id":"825501","model":"molmoact2-droid","state":"RUNNING"}
```

Active jobs are read from `squeue --json`; jobs that have already finished fall
back to `sacct --json`.

### Cancel a job

```bash
curl -X POST \
  -H "Authorization: Bearer $LAUNCHER_API_TOKEN" \
  "$URL/jobs/825501/cancel"
# {"job_id":"825501","status":"cancel_requested"}
```

Cancellation is refused unless the job is owned by the current user **and** is a
registered RoboColosseum policy job. Arbitrary Slurm jobs cannot be cancelled
through this API.

## 10. Multiple robots / concurrency

The contractor has multiple Franka robots, so the launcher **allows multiple
policy jobs at once** — there is no global "only one job" lock. Concurrency is
bounded by two configurable safeguards:

- `launcher.max_active_jobs` — global ceiling across all policy jobs.
- `policies.<model>.max_replicas` — per-model concurrent cap.

Before submitting, the launcher (under a process-level lock) checks both limits
and rejects with a clear, machine-readable error if either is exceeded:

- `409 global_limit_reached`
- `429 model_replica_limit_reached`

It never auto-cancels another job to make room. The launcher does **not** decide
which policy controls which Franka — that stays with Aaron's matchmaking/routing
infrastructure.

> **V1 assumes a single launcher process** (one Uvicorn worker + one in-process
> admission lock). Do not scale it to multiple workers without adding shared
> coordination.

## 11. Secrets & child-job isolation

Child GPU jobs are submitted with `sbatch --export=NONE`, so the policy worker
sets up its own environment (via `slurm/molmoact2.sh`) and does **not** inherit
launcher-only secrets such as `LAUNCHER_API_TOKEN` or ngrok credentials.

The launcher does not modify `slurm/molmoact2.sh` and never appends
`--enable-action`; the GPU worker keeps its safe dry-run default.

## 12. Stopping the launcher

```bash
scancel <launcher_job_id>
```

The Slurm `trap` in `launcher.slurm` tears down ngrok and Uvicorn cleanly.
**Stopping the launcher does NOT cancel running GPU policy jobs** — those keep
running under Slurm until they finish or are cancelled explicitly.

## 13. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Launcher job exits immediately | `LAUNCHER_API_TOKEN` not exported before `sbatch`, or ngrok binary not found/executable. Check `logs/launcher/slurm-<id>.err`. |
| No public URL in the log | ngrok could not authenticate/connect. Run `ngrok config add-authtoken ...`, then check `logs/launcher/ngrok-<id>.log`. |
| API returns `401 unauthorized` | Missing/incorrect `Authorization: Bearer` header, or token mismatch. |
| `422 unsupported_model` | Model name is not in the `policies` registry. Check spelling (`molmoact2-droid`). |
| `409 global_limit_reached` | `max_active_jobs` reached. Wait for jobs to finish or raise the limit in config. |
| `429 model_replica_limit_reached` | That model's `max_replicas` reached. |
| `502 slurm_error` on submit | `sbatch` failed. Check submit-host Slurm access and `logs/launcher/`. |
| Child job stuck `PENDING` | Normal Slurm queueing (GPU availability). Inspect with `squeue --me`. |
| Inspecting launcher logs | `logs/launcher/slurm-<id>.out` (stdout), `.err` (stderr), `ngrok-<id>.log`. |

## 14. Running the tests

All Slurm subprocess calls are mocked — the tests never submit real jobs:

```bash
python -m pytest job_launcher_server/tests/ -q
```
