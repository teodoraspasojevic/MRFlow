#!/bin/sh
# Bridge the platform's /weights mount into the paths predict.py expects, then hand over to
# predict.py as PID 1.
#
# Simpler than R2V-MR-Generation's entrypoint.sh: MRFlow's weights.zip holds exactly three
# directories (denoiser_ema/, flux-vae-f8-16ch/, <text-encoder>/), no file-vs-directory split, no
# per-arm cfg-scale derivation -- one checkpoint, one image. Every top-level entry of /weights is
# symlinked into /opt/app/models/<name>; predict.py's resolve_dir() finds them there by name.
set -eu

WEIGHTS_DIR="${FORITHMUS_WEIGHTS:-/weights}"
MODELS_DIR="${MRFLOW_MODELS_DIR:-/opt/app/models}"
OUTPUT_DIR="${FORITHMUS_OUTPUT:-/output}"
CHECKPOINT_DIR="${FORITHMUS_CHECKPOINT:-/checkpoint}"

# The platform guarantees the /output symlink, not the leaf directory (organizers' README).
mkdir -p "$MODELS_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "[entrypoint] FATAL: $WEIGHTS_DIR is not mounted. Submit with --weights weights.zip." >&2
    exit 1
fi

for path in "$WEIGHTS_DIR"/*; do
    [ -e "$path" ] || continue
    name=$(basename "$path")
    ln -sfn "$path" "$MODELS_DIR/$name"
done

echo "[entrypoint] models in $MODELS_DIR:"
ls -lL "$MODELS_DIR" 2>&1 | sed 's/^/[entrypoint]   /'
echo "[entrypoint] input mount ${FORITHMUS_INPUT:-/input}:"
ls -l "${FORITHMUS_INPUT:-/input}" 2>&1 | head -20 | sed 's/^/[entrypoint]   /'

# Sanity check + GPU-count detection in one Python startup, so a broken image says so immediately
# and legibly instead of failing deep inside model loading. torch.cuda.is_available() silently
# returning False (no libcudart, CPU base image, --gpus not wired) is the organizers' own
# documented first troubleshooting entry -- it would otherwise burn the whole time budget on CPU.
# Detecting GPU count here (rather than hardcoding 1) means predict.py's DDP support (see its
# ddp_setup()) is exercised automatically if a multi-GPU tier is ever used.
echo "[entrypoint] sanity check: critical imports + GPU visibility"
NPROC=$(python - <<'PYEOF' || exit 1
import sys
try:
    import numpy, torch, xformers, diffusers, transformers, nibabel
except ImportError as exc:
    print(f"[entrypoint] FATAL: critical import failed: {exc}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("[entrypoint] FATAL: torch.cuda.is_available() is False -- no GPU visible to this "
          "container. MRFlow's cross-attention has no CPU kernel (xformers), so generation needs "
          "a GPU; running on CPU would silently burn the whole time budget instead of failing "
          "loudly. Check --gpus / nvidia-container-toolkit wiring.", file=sys.stderr)
    sys.exit(1)
n = torch.cuda.device_count()
print(f"[entrypoint] sanity check OK: numpy {numpy.__version__}, torch {torch.__version__}, "
      f"{n} GPU(s) visible", file=sys.stderr)
print(n)
PYEOF
)
case "$NPROC" in ''|*[!0-9]*) echo "[entrypoint] FATAL: sanity check produced no GPU count" >&2; exit 1 ;; esac

# `exec` matters beyond tidiness: predict.py (as PID 1, directly or as torchrun's child) must
# receive the platform's SIGTERM so its handler runs, and /output is flushed before teardown.
if [ "$NPROC" -gt 1 ]; then
    echo "[entrypoint] $NPROC GPU(s) visible -- launching predict.py under torchrun"
    exec torchrun --standalone --nproc_per_node="$NPROC" /opt/app/predict.py
else
    echo "[entrypoint] launching predict.py"
    exec python /opt/app/predict.py
fi
