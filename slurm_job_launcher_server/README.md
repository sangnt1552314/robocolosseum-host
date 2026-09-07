# RoboColosseum Job Launcher (temporary control)

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

The launcher **never** touches camera streams, observations, actions, Colosseum
sessions or inference — it only starts and stops Slurm jobs.

## 2. Prerequisites

- NUS SoC Slurm access (`sbatch`/`squeue`/`sacct`/`scancel`).
- Python 3.12 (`py312` venv at `/home/n/ntasang/py312`).
- A free [ngrok](https://dashboard.ngrok.com) account + the ngrok binary.

Install the launcher deps:

```bash
source /home/n/ntasang/py312/bin/activate
pip install -r slurm_job_launcher_server/requirements.txt
```

## 3. Installing ngrok (no root)

ngrok is a single static binary — install it into your home dir without root
(Linux x86-64):

```bash
mkdir -p ~/bin && cd /tmp
# Verify the current URL for your arch at https://ngrok.com/download
curl -L -o ngrok.tgz \
  https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok.tgz && mv ngrok ~/bin/ngrok && chmod +x ~/bin/ngrok
~/bin/ngrok version
```

> Check `uname -m`: use `linux-amd64` for `x86_64`, `linux-arm64` for `aarch64`.
> Get the canonical link from the ngrok download page.

Default binary path is `/home/n/ntasang/bin/ngrok` (override with `NGROK_BIN` or
the `launcher.ngrok_bin` config key).

## 4. Configuring ngrok authentication

Store your authtoken once (from the dashboard):

```bash
~/bin/ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

ngrok writes it to `~/.config/ngrok/ngrok.yml`.

> **Never commit the authtoken or `ngrok.yml`.**

### Optional: auto-freeing the domain (ERR_NGROK_334)

The free plan gives **one** static domain, held by one agent at a time. If a
previous launcher crashed (e.g. `SIGKILL`), its agent may still hold the domain
and a new launcher fails with `ERR_NGROK_334`.

`launcher.sh` already retries past a brief lingering session and refuses to
start if another `colosseum-launcher` job is running. To also stop a truly
orphaned agent automatically, give it an **ngrok API key** (from
https://dashboard.ngrok.com/api-keys — separate from the authtoken):

```bash
# add to slurm_job_launcher_server/.env (git-ignored) or export before sbatch
export NGROK_API_KEY='<your-ngrok-api-key>'
```

When set, the launcher stops existing tunnel sessions before starting (best-
effort; the retry loop still runs). Without a key, clear a stuck endpoint at
https://dashboard.ngrok.com/agents or kill the stray `ngrok` process.

> Treat `NGROK_API_KEY` like a secret. `.env` is git-ignored.

## 5. Launcher API authentication

Every endpoint (including `/health`) requires `Authorization: Bearer <token>`,
read from `LAUNCHER_API_TOKEN` (never logged/returned; constant-time compare).

Generate a strong token:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Provide it **one** of two ways:

**A — export before `sbatch`:**

```bash
export LAUNCHER_API_TOKEN='<token>'
```

**B (recommended) — private `.env` file** (`launcher.sh` sources it at startup if
the var isn't already set):

```bash
printf 'LAUNCHER_API_TOKEN=%s\n' "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  > slurm_job_launcher_server/.env
chmod 600 slurm_job_launcher_server/.env
```

Default path `slurm_job_launcher_server/.env` (override with `LAUNCHER_ENV_FILE`);
an already-set env value wins. `.env` is git-ignored.

> Never hard-code, commit or print this token. Share it out-of-band.

## 6. Configuration

Copy the example config and edit if needed (the real file is git-ignored):

```bash
cp slurm_job_launcher_server/launcher.example.yaml slurm_job_launcher_server/launcher.yaml
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

From the project root, with `LAUNCHER_API_TOKEN` exported (or set in
`slurm_job_launcher_server/.env`):

```bash
sbatch slurm_job_launcher_server/launcher.sh
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

Multiple Franka robots → **multiple concurrent policy jobs** allowed (no global
"one job" lock). Two configurable limits bound it:

- `launcher.max_active_jobs` — global ceiling.
- `policies.<model>.max_replicas` — per-model cap.

Before submitting (under a process lock) both are checked, rejecting with
`409 global_limit_reached` or `429 model_replica_limit_reached`. It never
auto-cancels to make room, and does **not** decide which policy controls which
Franka — that stays with Aaron's routing.

> **V1 assumes a single launcher process** (one Uvicorn worker + one in-process
> lock). Don't scale to multiple workers without shared coordination.

## 11. Secrets & child-job isolation

Child GPU jobs run with `sbatch --export=NONE`, so they set up their own env
(via `slurm/molmoact2.sh`) and never inherit `LAUNCHER_API_TOKEN` or ngrok
creds. The launcher never modifies `slurm/molmoact2.sh` or appends
`--enable-action` — the worker keeps its dry-run default.

## 12. Stopping the launcher

```bash
scancel <launcher_job_id>
```

The `trap` in `launcher.sh` tears down ngrok and Uvicorn. **Stopping the
launcher does NOT cancel running GPU policy jobs** — they keep running until they
finish or are cancelled explicitly.

## 13. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Launcher job exits immediately | `LAUNCHER_API_TOKEN` not exported before `sbatch`, or ngrok binary not found/executable. Check `logs/launcher/slurm-<id>.err`. |
| No public URL in the log | ngrok could not authenticate/connect. Run `ngrok config add-authtoken ...`, then check `logs/launcher/ngrok-<id>.log`. |
| `ERR_NGROK_334` (endpoint already online) | Another agent holds the single free domain. Stop it: `scancel` the old launcher job, kill the stray `ngrok` process, or stop it at https://dashboard.ngrok.com/agents. Set `NGROK_API_KEY` (section 4) to have the launcher do this automatically. |
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
python -m pytest slurm_job_launcher_server/tests/ -q
```
