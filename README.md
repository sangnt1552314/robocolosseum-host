# robocolosseum-hosted

Modular **policy hosting** for the FrodoBots Colosseum. The contractor runs
routing, matchmaking and scoring; **we host only policy inference** as a GPU job
(PBS on NUS Hopper, SLURM on NUS SoC). Supported: `allenai/MolmoAct2-DROID` and
`lerobot/pi05_droid`.

## Architecture

```
Franka / DROID robot
        ↓
FrodoBots Colosseum Router
        ↓  outbound WSS (SDK, no public server on our side)
PBS GPU worker on Hopper  ── runner.py (policy-independent lifecycle)
        ↓
Policy adapter (BasePolicyAdapter)
        ↓
MolmoAct2-DROID  →  action chunk (15 × 8)
        ↓
Colosseum SDK  →  Router / robot   (only with --enable-action)
```

The compute node opens an **outbound** WebSocket to the router using the
official [`colosseum-policy-server`](https://github.com/frodobots-org/colosseum-policy-server)
SDK. We never expose a public HTTP server.
## Layout

```
configs/    example policy + router YAML (real router.yaml is git-ignored)
pbs/        PBS submission scripts (one per model environment) — NUS Hopper
slurm/      SLURM submission scripts (one per model environment) — NUS SoC
scripts/    run_policy.py + the two read-only test scripts
src/robocolosseum/
    config.py       YAML + env-var configuration
    runner.py       connect → observe → predict → validate → (send|print)
    policies/
        base.py     BasePolicyAdapter interface
        registry.py name → adapter (lazy import, keeps envs isolated)
        molmoact2.py
    utils/          timing + action validation
```

## Setup

No packaging step — scripts add `src/` to `sys.path`. Install deps into the
model's environment and run directly.

1. **Colosseum SDK** (no PyPI release):

   ```bash
   pip install "colosseum-policy-server @ git+https://github.com/frodobots-org/colosseum-policy-server"
   ```

2. **Model deps** — MolmoAct2: `pip install torch transformers pillow numpy pyyaml`
   (add `opencv-python` only if the router streams JPEG/PNG). pi05_droid:
   `pip install lerobot`.

3. **Use the existing HF cache** (checkpoint already on Hopper):

   ```bash
   export HF_HOME=/scratch/e1583535/cache
   export HF_HUB_OFFLINE=1
   ```

4. **Router credentials** (never commit). Env vars:

   ```bash
   export COLOSSEUM_ROUTER_URL="wss://router.example.com:8443"
   export COLOSSEUM_TOKEN="pol_..."
   ```

   or copy `configs/router.example.yaml` → `configs/router.yaml` (git-ignored,
   `chmod 600`) and pass `--router-config configs/router.yaml`.

5. **Policy config**: `cp configs/molmoact2.example.yaml configs/molmoact2.yaml`
   (or `configs/pi05_droid.example.yaml`).

## Running the milestone tests (in order)

```bash
# 1. Mock inference — no router/robot. Loads the checkpoint, produces an action.
python scripts/test_molmo_mock.py --config configs/molmoact2.yaml

# 2. Read-only router — inspect a REAL observation. Never sends an action.
python scripts/test_router_readonly.py

# 3. Dry run — Router → model → printed action (nothing sent).
python scripts/run_policy.py --policy molmoact2 --config configs/molmoact2.yaml

# 4. Move the robot (only when ready):
python scripts/run_policy.py --policy molmoact2 --config configs/molmoact2.yaml --enable-action
```

## Submit the PBS job (NUS Hopper)

```bash
qsub pbs/molmoact2.pbs
```

Runs inside the Hopper PyTorch singularity image (`pytorch_2.6.0_cuda_12.8.sif`)
and activates `/scratch/e1583535/virtualenvs/robocolosseum`. Set router
credentials before `qsub`; confirm the PBS project and resource line.

## Submit the SLURM job (NUS SoC)

```bash
sbatch slurm/molmoact2.sh        # or slurm/pi05_droid.sh
```

Runs **natively** (no singularity), activates `py312` at
`/home/n/ntasang/py312`, caches HF models at `/home/n/ntasang/cache`. Set router
credentials before `sbatch`; confirm `--gres`, `ENV_NAME`, `HOME_PATH`.

> **Note:** `--time` (like PBS `walltime`) is a safety cap only — the worker
> exits and the GPU is released as soon as the session finishes or a
> startup/idle timeout fires.

## Safety

> **No actions are sent by default** — the worker only prints and validates the
> action chunk. Sending requires the explicit `--enable-action` flag.

Every chunk is validated (finite, correct dims/`action_dim`, within horizon)
before it could be sent; the SDK re-validates on `send_action`.

## GPU release

`walltime` / `--time` is a **maximum**, not a target. The worker exits — freeing
the GPU — as soon as **any** of these happen: the session completes, the startup
timeout elapses with no session, the idle timeout elapses with no new
observation, or an unrecoverable error occurs. No `sleep` loops keep it alive;
tune `runtime.idle_timeout_seconds`.

### Session semantics (verified against the SDK)

The SDK is pull-based: `sdk.get_obs(timeout=...)` returns the latest
observation, raising `TimeoutError` on timeout and `SDKConnectionError` on a
disconnect. `RESET` and `SESSION_CLOSE` are **not** delivered through
`get_obs`. The runner therefore:

* treats a change in `observation.session_id` as a new session/rollout and
  calls `adapter.reset()`;
* treats "timeout with an empty `sdk.session_id`" (or a disconnect after a
  session started) as a finished evaluation and exits.

## Adding a new policy

The common runner, PBS lifecycle, timeout logic, logging and action-send switch
do **not** need to change. To add e.g. `OpenGalaxea/G05`, `lerobot/pi05_droid`,
`lihzha/LAP-3B`, or `nvidia/GR00T-N1.7-DROID`:

1. Create `src/robocolosseum/policies/my_policy.py` implementing
   `BasePolicyAdapter` (`load`, `predict`, optional `reset`/`close`).
2. Register it in `src/robocolosseum/policies/registry.py`:
   ```python
   POLICY_REGISTRY = {
       "molmoact2": "robocolosseum.policies.molmoact2:MolmoAct2Adapter",
       "my_policy": "robocolosseum.policies.my_policy:MyPolicyAdapter",
   }
   ```
   (Adapters are imported lazily, so each model can keep its own environment.)
3. Add `configs/my_policy.yaml`.
4. Add a submission script that activates that model's environment and runs
   `scripts/run_policy.py --policy my_policy`:
   * `pbs/my_policy.pbs` for NUS Hopper, and/or
   * `slurm/my_policy.sh` for NUS SoC.
5. Run the standard dry-run tests.

### Example: `lerobot/pi05_droid` (already wired)

* Adapter `src/robocolosseum/policies/pi05_droid.py` (`Pi05Adapter`) on the
  LeRobot `PI0Policy` API (`from_pretrained`, `make_pre_post_processors`,
  `predict_action_chunk`). The processor normalises inputs and tokenises the
  instruction; the post-processor un-normalises actions.
* Registered `"pi05_droid"`; config `configs/pi05_droid.example.yaml`; script
  `slurm/pi05_droid.sh` (`colosseum-pi05`); launcher entry `pi05-droid`.
* Install `lerobot` in that env first. Checkpoint specifics already set in the
  config: image keys `base_0_rgb` / `left_wrist_0_rgb`, and `action_dim: 8`
  (pi0.5 pads actions to 32 — only the first 8 DROID dims are used). **Verify**
  the camera mapping and state layout against a live observation.

## Unverified assumptions

* **Camera mapping** (`external: left_image`, `wrist: right_image`) is a
  placeholder. Verify field names/shapes with `test_router_readonly.py` against
  a real DROID observation and update `configs/molmoact2.yaml`.
* **State layout**: the adapter builds the 8-dim state as
  `joints (7) + gripper (1)`. Confirm the joint count and gripper convention of
  the live robot match the DROID checkpoint.
* **PBS resource syntax** in `pbs/molmoact2.pbs` must be confirmed for Hopper.
