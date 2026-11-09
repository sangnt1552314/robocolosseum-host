#!/bin/bash
# One-time setup for serving on NSCC (login node: needs internet, no GPU). Idempotent: rerun any
# step that fails. Reuses the venvs and repos built by robocolosseum-training/scripts/nscc/setup_envs.sh
# and adds only what inference needs. HF_TOKEN comes from the environment, this folder's .env or the
# training repo's .env.
#
#   bash nscc_pbs_remote_inference/pbs/setup.sh [all|models|pi05|tailscale|check ...]
#
#   models     pinned inference checkpoints -> $HF_HUB_CACHE, and the pinned G05 assets -> $G05_ASSETS
#   pi05       colosseum-pi05 venv: LeRobot b6ec006 + the patch shipped with hqfang/pi05-so100_101
#   tailscale  tailscale + tailscaled binaries -> $TOOLS_ROOT/tailscale
#   check      report any venv or path named in the so101 and yam configs that is missing
#
# franka is not downloaded by default. To serve it, download allenai/MolmoAct2-DROID at the revision
# in configs/franka/molmoact2.yaml.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PI05_SO101_REV=d3204e03ae84d232e2493a933fd460515904eb58
LEROBOT_PI05_SHA=b6ec0060779550c0a157ae34feb89e0cf86012a8
TAILSCALE_VERSION=1.104.5
G05_REV=e312be81e90c56a55bcb26b57429bd39a335b449

load_hf_token() {
    local env_file
    for env_file in "$INFER_ENV_FILE" "$TANSANG_ROOT/projects/robocolosseum-training/.env"; do
        if [ -z "${HF_TOKEN:-}" ] && [ -f "$env_file" ]; then
            HF_TOKEN=$(grep -E '^HF_TOKEN=' "$env_file" | cut -d= -f2- | tr -d "\"'" || true)
        fi
    done
    export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not found: the RoboColosseum repos are private}"
}

step_models() {
    load_hf_token
    export HF_XET_HIGH_PERFORMANCE=1
    local hf="$VENVS_ROOT/hftools/bin/hf"

    "$hf" download allenai/MolmoAct2-SO100_101 --revision 152569fe57914d97be91055800035f54e250d009
    "$hf" download allenai/MolmoAct2-BimanualYAM --revision 8dcbed66f2380e4393189c303ea72488eb9e63c2
    "$hf" download hqfang/pi05-so100_101 --revision "$PI05_SO101_REV"
    "$hf" download NUSMAGIC/pi05-MolmoAct2-YAM --revision 2f28d00ac28c543626f5a4f579a3da09bee4a4ed
    "$hf" download hqfang/groot-so100_101 --revision f5e17af8948a647e4c23186a27927bf73d6d784e
    "$hf" download NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM --revision 5ba9c403bdb962b88c39f843da536eb8370b61b2
    "$hf" download helen9975/lingbot-vla-v2-6b-molmoact2-yam --revision af7529f2ea27ff6e77911a98f86c40cfb75d229e
    "$hf" download RoboColosseum/LingBot-MolmoAct2-So101 --revision af0bbb684bf0579d1c4ba19cfd11b8ebd0de1e18 \
        --exclude "checkpoint-90000/*" \
        --exclude "checkpoint-100000/finetune_assets/so100_robots/*"   # training only, 1215 files
    "$hf" download RoboColosseum/G05-MolmoAct2-YAM --revision 26c91270ce33325a29990b3ff7ea7a4b47bd4ec1

    # Base models and tokenizers. Qwen3-VL is used for its processor files only.
    "$hf" download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495
    "$hf" download google/paligemma-3b-pt-224 --revision 35e4f46485b4d07967e7e9935bc3786aad50687c
    "$hf" download Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
        --exclude "*.safetensors"
    "$hf" download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
        --exclude "*.safetensors"

    # Pinned to the revision the configs name, and kept apart from training's projects/GalaxeaVLA/checkpoints.
    mkdir -p "$G05_ASSETS"
    "$hf" download OpenGalaxea/G05 --revision "$G05_REV" --local-dir "$G05_ASSETS" \
        --include "g05-so101/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*"
}

step_pi05() {
    local dst="$DATA_ROOT/deps/lerobot-pi05-so101"
    local patch="$HF_HUB_CACHE/models--hqfang--pi05-so100_101/snapshots/$PI05_SO101_REV/code/lerobot.patch"
    [ -f "$patch" ] || { echo "ERROR: $patch not found; run the models step first." >&2; exit 1; }

    # Everything this step writes stays on /data: this venv is tens of thousands of files, and
    # scratch has a file-count limit.
    local UV_CACHE_DIR="$DATA_ROOT/cache/uv" TMPDIR="$DATA_ROOT/tmp"
    export UV_CACHE_DIR TMPDIR PYTHONDONTWRITEBYTECODE=1
    mkdir -p "$(dirname "$dst")" "$(dirname "$PI05_VENV")" "$UV_CACHE_DIR" "$TMPDIR"
    [ -d "$dst/.git" ] || git clone --filter=blob:none https://github.com/huggingface/lerobot.git "$dst"
    git -C "$dst" checkout -q "$LEROBOT_PI05_SHA"
    if git -C "$dst" diff --quiet; then
        git -C "$dst" apply "$patch"
    fi

    [ -x "$PI05_VENV/bin/python" ] || uv venv -q -p 3.12 "$PI05_VENV"
    uv pip install -p "$PI05_VENV" --torch-backend cu128 -e "$dst[pi]"
    # As in training: the +cu128 torchcodec wheel needs NVIDIA NPP, which no venv here ships.
    local v
    v="$("$PI05_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("torchcodec").split("+")[0])' 2>/dev/null || true)"
    if [ -n "$v" ]; then
        uv pip install -p "$PI05_VENV" --index-url https://pypi.org/simple --no-deps --reinstall "torchcodec==$v"
    fi
    "$PI05_VENV/bin/python" -c "import torch, lerobot; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; \
print('lerobot', lerobot.__version__, 'torch', torch.__version__, torch.version.cuda)"
}

step_tailscale() {
    local name="tailscale_${TAILSCALE_VERSION}_amd64"
    mkdir -p "$TOOLS_ROOT/tailscale"
    curl -fsSL "https://pkgs.tailscale.com/stable/$name.tgz" \
        | tar -xz --strip-components=1 -C "$TOOLS_ROOT/tailscale" "$name/tailscale" "$name/tailscaled"
    "$TOOLS_ROOT/tailscale/tailscale" version | head -1
}

step_check() {
    local missing=0 path
    for path in "$LEROBOT_VENV/bin/python" "$LINGBOT_VENV/bin/python" "$G05_VENV/bin/python" "$PI05_VENV/bin/python" \
                "$TOOLS_ROOT/tailscale/tailscaled"; do
        [ -e "$path" ] || { echo "MISSING $path"; missing=$((missing + 1)); }
    done
    while read -r path; do
        [ -e "$path" ] || { echo "MISSING $path"; missing=$((missing + 1)); }
    done < <(grep -hvE '^\s*#' "$HOST_ROOT/$INFER_DIR"/configs/{so101,yam}/*.yaml | grep -oE "($TANSANG_ROOT|$DATA_ROOT)/[^ ]+" | sort -u)
    [ -s "$INFER_ENV_FILE" ] && grep -q '^TS_AUTHKEY=.' "$INFER_ENV_FILE" || echo "NOTE: TS_AUTHKEY is not set in $INFER_ENV_FILE yet"
    echo "check: $missing missing"
    [ "$missing" -eq 0 ]
}

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(all)
if [ "${STEPS[0]}" = all ]; then
    STEPS=(models pi05 tailscale check)
fi
for s in "${STEPS[@]}"; do
    echo "==================== setup: $s ===================="
    "step_$s"
done
echo "Setup done: ${STEPS[*]}"
