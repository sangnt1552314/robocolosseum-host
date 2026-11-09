# Shared paths for serving on NSCC ASPIRE 2B (sourced by the scripts in this folder). Same layout as
# robocolosseum-training/scripts/nscc/env.sh, without the training-only secrets and W&B settings.
# No Singularity here: each model family runs directly from its uv virtualenv.

export TANSANG_ROOT=/scratch/users/nus/duanj1/tansang
export HOST_ROOT="$TANSANG_ROOT/projects/robocolosseum-host"
export INFER_DIR="nscc_pbs_remote_inference"     # relative to $HOST_ROOT
export INFER_ENV_FILE="$HOST_ROOT/$INFER_DIR/.env"   # git-ignored: TS_AUTHKEY, optional ngrok settings
export TOOLS_ROOT="$TANSANG_ROOT/tools"
export DATA_ROOT=/data/projects/51001013/tansang     # models, logs, caches: no file-count limit, visible on GPU nodes
export LOG_ROOT="$DATA_ROOT/logs/$INFER_DIR"
export G05_ASSETS="$DATA_ROOT/cache/G05-e312be81"     # OpenGalaxea/G05 at the revision the configs pin

export VENVS_ROOT="$TANSANG_ROOT/venvs"
export LEROBOT_VENV="$VENVS_ROOT/lerobot-v0.6.1"  # shared with training, used read-only
export LINGBOT_VENV="$VENVS_ROOT/lingbot-vla-v2"  # shared with training, used read-only
export G05_VENV="$VENVS_ROOT/g05"                 # shared with training, used read-only
export PI05_VENV="$DATA_ROOT/venvs/colosseum-pi05"  # inference only, built by setup.sh; on /data for the file-count limit

export PATH="$TANSANG_ROOT/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TANSANG_ROOT/python"
export UV_CACHE_DIR="$TANSANG_ROOT/cache/uv"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

export HF_HOME="$DATA_ROOT/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
# Jobs write everything below to /data: scratch is at its file-count limit, and a full cache crashes a model.
export TMPDIR="$DATA_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$DATA_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/cache/inductor"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"
export MPLCONFIGDIR="$DATA_ROOT/cache/matplotlib"

export FFMPEG_ROOT="$TANSANG_ROOT/opt/ffmpeg-7.1"
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$TMPDIR" "$LOG_ROOT" "$DATA_ROOT/logs/pbs" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# GPUs allocated to this job, one per line. PBS may or may not set CUDA_VISIBLE_DEVICES; nvidia-smi
# only lists the job's own GPUs on these nodes.
nscc_gpu_list() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES"
    else
        nvidia-smi --query-gpu=uuid --format=csv,noheader
    fi
}

# First model port for an embodiment. g1 jobs share nodes, so the so101 and yam jobs can land on the
# same node; separate port ranges keep them from colliding on 127.0.0.1.
nscc_base_port() {  # nscc_base_port <robot>
    case "$1" in
        so101)  echo 9101 ;;
        yam)    echo 9111 ;;
        franka) echo 9121 ;;
        *) echo "ERROR: unknown embodiment '$1' (so101, yam, franka)" >&2; return 1 ;;
    esac
}
