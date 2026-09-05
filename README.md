# robocolosseum-hosted

Modular robotics **policy hosting** for the FrodoBots Colosseum, running as a
PBS GPU job on NUS Hopper. The contractor hosts routing, matchmaking and
scoring; **we host only policy inference**. The first supported policy is
`allenai/MolmoAct2-DROID`.

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

There is no packaging step yet — the scripts add `src/` to `sys.path`
themselves, so just install the dependencies into the model's environment and
run the scripts directly.

1. **Install the Colosseum SDK** (no PyPI release):

   ```bash
   pip install "colosseum-policy-server @ git+https://github.com/frodobots-org/colosseum-policy-server"
   ```

2. **Install the MolmoAct2 dependencies** in the same environment:

   ```bash
   pip install torch transformers pillow numpy pyyaml
   # Add opencv-python only if the router streams JPEG/PNG frames (not RAW_RGB).
   ```

3. **Use the existing Hugging Face cache** (the checkpoint is already on Hopper
   at `/scratch/e1583535/cache`):

   ```bash
   export HF_HOME=/scratch/e1583535/cache
   export HF_HUB_OFFLINE=1
   ```

4. **Configure the router credentials** (never commit them). Either:

   ```bash
   export COLOSSEUM_ROUTER_URL="wss://router.example.com:8443"
   export COLOSSEUM_TOKEN="pol_..."
   ```

   or copy `configs/router.example.yaml` to `configs/router.yaml` (git-ignored),
   `chmod 600 configs/router.yaml`, and pass `--router-config configs/router.yaml`.

5. **Copy the policy config**:

   ```bash
   cp configs/molmoact2.example.yaml configs/molmoact2.yaml
   ```

## Running the milestone tests (in order)

```bash
# 1. Mock inference — no router, no robot. Loads the checkpoint and produces
#    an action chunk from fake DROID-style input.
python scripts/test_molmo_mock.py --config configs/molmoact2.yaml

# 2. Read-only router — inspect a REAL observation. Never sends an action.
python scripts/test_router_readonly.py

# 3. Integrated dry run — Router → MolmoAct2 → printed action (nothing sent).
python scripts/run_policy.py --policy molmoact2 --config configs/molmoact2.yaml

# 4. Only when you are ready to actually move the robot:
python scripts/run_policy.py --policy molmoact2 --config configs/molmoact2.yaml --enable-action
```

## Submit the PBS job (NUS Hopper)

```bash
qsub pbs/molmoact2.pbs
```

The script runs the worker inside the Hopper PyTorch singularity image
(`pytorch_2.6.0_cuda_12.8.sif`) and activates
`/scratch/e1583535/virtualenvs/robocolosseum`. Set your router credentials in
the environment (or a private sourced file) before `qsub`. Confirm the PBS
project (`CFP01-CF-002`) and resource line for your account.

## Submit the SLURM job (NUS SoC)

```bash
sbatch slurm/molmoact2.sh
```

Unlike the Hopper PBS worker, the SoC script runs **natively** (no singularity)
and activates the `py312` virtualenv at `/home/n/ntasang/py312`. Hugging Face
model downloads are cached persistently at `/home/n/ntasang/cache` so they
survive across jobs. Set your router credentials in the environment (or a
private sourced file) before `sbatch`. Confirm the `--gres` GPU type
(`h100-47:1`), `ENV_NAME`, and `HOME_PATH` for your account.

> **Note:** `--time` (like PBS `walltime`) is a safety cap only — the worker
> exits and SLURM releases the GPU as soon as the session finishes or a
> startup/idle timeout fires.

## Safety

> **By default, no actions are sent to the robot.** The worker prints and
> validates the action chunk only.
>
> Actual robot actions require the explicit flag:
> ```
> --enable-action
> ```

Every action chunk is validated (finite, correct dimensionality, correct
`action_dim`, within the horizon limit) before it could ever be sent; the SDK
re-validates on `send_action`.

## PBS behavior (GPU release)

`walltime` is the **maximum** allowed duration, not a target. The worker exits
— and PBS releases the GPU — as soon as **any** of these happens:

* the evaluation session completes,
* the startup timeout elapses with no session,
* the idle timeout elapses with no new observation,
* an unrecoverable error occurs (non-zero exit).

There are no artificial `sleep` loops keeping the job alive. Tune
`runtime.idle_timeout_seconds` to control how quickly the worker gives up after
the last observation.

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

## Unverified assumptions

* **Camera mapping** (`external: left_image`, `wrist: right_image`) is a
  placeholder. Verify field names/shapes with `test_router_readonly.py` against
  a real DROID observation and update `configs/molmoact2.yaml`.
* **State layout**: the adapter builds the 8-dim state as
  `joints (7) + gripper (1)`. Confirm the joint count and gripper convention of
  the live robot match the DROID checkpoint.
* **PBS resource syntax** in `pbs/molmoact2.pbs` must be confirmed for Hopper.
