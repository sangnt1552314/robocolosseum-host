# pbs_remote_inference

Serve VLA models on Hopper H200 GPUs for
[Colosseum Policy Server](https://github.com/frodobots-org/colosseum-policy-server).
Each embodiment runs on one GPU with 5 models, reached over Tailscale. Run all Hopper
commands from `/scratch/e1583535/projects/robocolosseum-hosted`, on the normal shell. Do
**not** start a singularity shell; the scripts start each model in its own container.

## 1. Start the models on Hopper

**Batch job** (recommended for evaluations; default walltime 2 h, change with `-l walltime=08:00:00`):

```bash
qsub -v EMBODIMENT=so101,MODELS=molmoact2:pi05:groot:g05:lingbot,TUNNEL=tailscale pbs_remote_inference/pbs/cohost.pbs
qsub -v EMBODIMENT=yam,MODELS=molmoact2:pi05:groot:lingbot:g05,TUNNEL=tailscale pbs_remote_inference/pbs/cohost.pbs
```

**Interactive job** (quick tests). It ends when your terminal closes, so use `tmux` for long runs:

```bash
qsub -I -P CFP05-SF-002 -l select=1:ngpus=1 -l walltime=02:00:00
cd /scratch/e1583535/projects/robocolosseum-hosted
bash pbs_remote_inference/pbs/interactive.sh so101 molmoact2:pi05:groot:g05:lingbot tailscale
# or, in a second interactive job:
bash pbs_remote_inference/pbs/interactive.sh yam molmoact2:pi05:groot:lingbot:g05 tailscale
```

The script prints `READY <model>` per model after 1–5 min; Ctrl-C stops everything.

**Check / stop a batch job:**

```bash
qstat -u $USER                                                         # R = running
cat logs/pbs_remote_inference/tailscale/status.<jobid>.txt                # tailnet node is up
grep -h Serving logs/pbs_remote_inference/so101/*/stdout.<jobid>.log      # one line per ready model
qdel <jobid>                                                           # stop
```

`<jobid>` is the full ID `qsub` printed, e.g. `706224.hopper-m-02`. Logs are in
`logs/pbs_remote_inference/<robot>/<model>/`.

## 2. Connect the Policy Server machine

On the Policy Server machine (once): install Tailscale and log in to the same tailnet,
then copy `pbs_remote_inference/policy_server_relay.py` there.

```bash
tailscale status          # colosseum-so101 and colosseum-yam should be listed
# relays: the Policy Server only accepts http://127.0.0.1:<port> endpoints
for i in 1 2 3 4 5; do python3 policy_server_relay.py --remote http://colosseum-so101:910$i --port 920$i & done
for i in 1 2 3 4 5; do python3 policy_server_relay.py --remote http://colosseum-yam:910$i   --port 921$i & done
```

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

## First-time setup

- `pbs_remote_inference/.env` (git-ignored) needs `TS_AUTHKEY=tskey-auth-...`, a
  reusable + ephemeral Tailscale auth key.
- Checkpoints must already be in `/scratch/e1583535/cache`; jobs run offline.

Everything else (models, environments, measurements, other tunnels, HTTP contract,
adding a model) is in [DETAILS.md](DETAILS.md).
