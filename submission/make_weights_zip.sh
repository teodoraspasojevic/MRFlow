#!/usr/bin/env bash
# Assemble the weights.zip that `forithmus submit --weights` mounts at /weights.
#
#     ./submission/make_weights_zip.sh                      # checkpoint-60000, default paths
#     ./submission/make_weights_zip.sh checkpoint-45000      # a different checkpoint
#
#     # a checkpoint trained with the sectioned conditioning needs all three of its encoders:
#     ENCODERS="BiomedVLP-CXR-BERT-specialized MedEmbed-large-v0.1 Bio_ClinicalBERT" \
#         ./submission/make_weights_zip.sh checkpoint-100000
#
# Layout produced (files at the ZIP ROOT -- a parent folder breaks the platform's mount, exactly as
# in R2V-MR-Generation/submission/make_weights_zip.sh):
#
#     denoiser_ema/                       the fine-tuned STDiT checkpoint (config.json + safetensors)
#     flux-vae-f8-16ch/                   frozen FLUX VAE
#     BiomedVLP-CXR-BERT-specialized/     frozen text encoder (one directory per $ENCODERS entry)
#
# entrypoint.sh symlinks each of these into /opt/app/models/<name>; predict.py's resolve_dir()
# finds them there by name, and needs no manifest or arm file -- there is exactly one arm.
set -euo pipefail

CKPT="${1:-checkpoint-60000}"
WS="${WS:-/vol/idea_ramses/va47zasy/VLM3D-MICCAI-2026}"
OUT_DIR="${OUT_DIR:-${WS}/submission/weights}"

DENOISER="${WS}/models/mrflow/${CKPT}/denoiser_ema"
VAE="${WS}/models/flux-vae-f8-16ch"
# Which frozen text encoders the checkpoint's conditioning needs -- one for cxr_bert_cls, three for
# report2ct_style_meta (see echosyn/common/mrrate.py CONDITIONINGS). predict.py resolves the
# encoder root as the CXR-BERT directory's parent, which is where entrypoint.sh symlinks all of
# them, so the pooled name has to stay in the list.
ENCODERS="${ENCODERS:-BiomedVLP-CXR-BERT-specialized}"
TEXT_ENCODER_ROOT="${WS}/models/text-encoders"

for d in "$DENOISER" "$VAE"; do
    [ -d "$d" ] || { echo "missing: $d" >&2; exit 1; }
done
for name in $ENCODERS; do
    [ -d "${TEXT_ENCODER_ROOT}/${name}" ] || { echo "missing: ${TEXT_ENCODER_ROOT}/${name}" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
echo "staging $CKPT in $STAGE"

mkdir -p "$STAGE/denoiser_ema" "$STAGE/flux-vae-f8-16ch"
cp -L "$DENOISER"/config.json "$DENOISER"/diffusion_pytorch_model.safetensors "$STAGE/denoiser_ema/"
cp -L "$VAE"/config.json "$VAE"/diffusion_pytorch_model.safetensors "$STAGE/flux-vae-f8-16ch/"
for name in $ENCODERS; do
    mkdir -p "$STAGE/$name"
    # Weight + tokenizer + config files only -- a pretrained snapshot directory can also carry a
    # .git, a README and other upload-cost-only files (same rationale as R2V-MR-Generation's own
    # script).
    find "${TEXT_ENCODER_ROOT}/${name}" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.txt' -o -name '*.safetensors' \) \
        -exec cp -L {} "$STAGE/$name/" \;
    if ! ls "$STAGE/$name"/*.safetensors >/dev/null 2>&1; then
        echo "FATAL: no .safetensors in ${TEXT_ENCODER_ROOT}/${name}, only a legacy" >&2
        echo "pytorch_model.bin -- this would build a zip that fails to load under" >&2
        echo "transformers>=4.51 (torch.load's CVE-2025-32434 guard on torch<2.6)." >&2
        echo "Convert it to model.safetensors first." >&2
        exit 1
    fi
done

# rsync from Helma preserves the source's own mode bits, and checkpoint-60000/denoiser_ema's
# safetensors file came across as `-rw-------` (owner-only) -- fine on this host, but the platform
# (and this script's own docker-run test) runs the container as a non-root `appuser` that doesn't
# match whichever uid unzips /weights, so an owner-only file inside the zip is unreadable there.
# Confirmed reproducible: `PermissionError` loading denoiser_ema/diffusion_pytorch_model.safetensors
# under `docker run` before this normalization was added.
find "$STAGE" -type f -exec chmod 644 {} +
find "$STAGE" -type d -exec chmod 755 {} +

mkdir -p "$OUT_DIR"
ZIP="${OUT_DIR}/mrflow_weights.zip"
rm -f "$ZIP"
# -0 (store, no deflate): the .safetensors files are already-compressed tensors, so deflate spends
# minutes to save single-digit percent, and the platform unzips this on every run.
( cd "$STAGE" && zip -r0 -q "$ZIP" . -x '.*' )

echo
echo "wrote $ZIP  ($(du -h "$ZIP" | cut -f1))"
echo "contents (must show no parent-directory prefix):"
unzip -l "$ZIP" | sed 's/^/  /'
echo
echo "next:  forithmus submit mrflow_submission.tar.gz --phase <phase> --tier gpu-a100-80 \\"
echo "           --time-budget 240 --weights $ZIP -d \"MRFlow STDiT-L (60k-step MR-RATE fine-tune)\""
