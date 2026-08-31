# VLM3D `mr-volume-generation` submission -- MRFlow (STDiT + flow matching, autoregressive).
# Thin image: code + baked config/native-spacing table only, weights mounted at /weights.
#
#     docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mrflow-vlm3d-challenge:latest .
#
# `--platform linux/amd64` is not optional if you build on an Apple-silicon Mac: the platform's
# validator rejects anything that is not amd64, and it rejects it after you have paid for the
# upload (see R2V-MR-Generation/submission/SUBMISSION_LESSONS.md).
#
# Why a CUDA base and not python:3.12-slim: the slim images carry no libcudart, so
# torch.cuda.is_available() silently returns False and the run burns its whole time budget on CPU.
# MRFlow's cross-attention (xformers' memory_efficient_attention) has no CPU kernel either way, so
# this must fail loudly instead -- see entrypoint.sh's sanity check.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bzip2 ca-certificates \
        libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Several challenge platforms in this family (see R2V-MR-Generation's own lessons)
# refuse an image whose effective USER is root.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Miniconda for Python 3.12, matching the interpreter MRFlow was trained and evaluated under
# (3.12.3, this repo's .venv). Ubuntu 22.04's system Python is 3.10.
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh \
        -o /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p /opt/conda \
    && rm /tmp/conda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Versions pinned to the exact set MRFlow's own dev environment resolved (MRFlow/.venv, python
# 3.12.3, cu128) -- see the CLAUDE.md gotcha on VAE/init_from sensitivity: floating any of these
# is a different environment than the one the checkpoint was produced and locally evaluated under.
# torch/torchvision/xformers must come from the same cu128 build line, hence one --index-url call.
RUN pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 xformers==0.0.35 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
        diffusers==0.39.0 \
        transformers==5.15.1 \
        tokenizers==0.22.2 \
        safetensors==0.8.0 \
        huggingface-hub==1.28.0 \
        accelerate==1.14.0 \
        wandb==0.28.2 \
        torchdiffeq==0.2.5 \
        einops==0.8.2 \
        timm==1.0.28 \
        omegaconf==2.3.1 \
        nibabel==5.4.2 \
        numpy==2.5.2 \
        scipy==1.18.0 \
        opencv-python-headless==5.0.0.93 \
        imageio==2.37.4 \
        imageio-ffmpeg==0.6.0 \
        Pillow==11.3.0 \
        pyarrow \
        tqdm \
    && conda clean -afy \
    && (find /opt/conda -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true)

# No network egress at runtime. Every checkpoint arrives through /weights, so nothing should ever
# reach for the Hub -- these make that a loud failure instead of a 30s DNS hang per encoder.
# WANDB_MODE=disabled: echosyn.common imports wandb at module load (train.py's tracker), which
# predict.py never calls into, but the import itself must not try to phone home or prompt a login.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The in-container tree:
#
#   /opt/app/echosyn/                   MRFlow's shared package (denoiser, VAE scaling, mrrate.py)
#   /opt/app/auto_regressive_generate/  LatentAutoregressiveGenerator
#   /opt/app/{predict.py,entrypoint.sh,native_spacing.py}
#   /opt/app/config.yaml                 the training config saved next to the checkpoint, with
#                                         vae.pretrained/mri.text_checkpoint overridden at runtime
#                                         by predict.py once the weight dirs below are resolved
#   /opt/app/native_spacing_table.json    MR-RATE's own slice-thickness survey (model-agnostic)
#   /opt/app/models/                     -> symlinks into /weights (made by entrypoint.sh):
#                                            denoiser_ema/, flux-vae-f8-16ch/, <text-encoder>/
ENV PYTHONPATH=/opt/app \
    MRFLOW_MODELS_DIR=/opt/app/models

# Submission-time knobs, baked in rather than left to predict.py's own fallback defaults. The real
# platform passes no runtime flags or env vars at all -- `docker run -e ...` only ever works for
# local testing -- so once these are here, the only participant-controlled input left at
# container-run time is which weights.zip is mounted at /weights. Change a value and rebuild.
#
# MRFLOW_MODALITY_CFG_SCALE / MRFLOW_REPORT_CFG_SCALE: 1.0/1.0 (config.guidance's own default) is
# the plain conditional prediction -- one denoiser call per Euler step instead of three. A future
# submission sweeps these (see auto_regressive_generate/main.py's own --modality-cfg-scale /
# --report-cfg-scale for what a value other than 1.0 does).
#
# MRFLOW_NATIVE_SPACING_MODE=sample: slab-averages each finished volume's slice axis down to a
# thickness drawn from MR-RATE's own per-(modality, plane) distribution before writing it -- see
# native_spacing.py. This is what R2V-MR-Generation's own Dockerfile.native_spacing measured as a
# large FID win, for the same reason it applies here: FID_2p5D never resamples, so a generated
# volume's always-~1mm slice grid is otherwise scored against native 4-6.5mm 2D acquisitions on a
# mismatched grid. Set to "off" to write the generated grid unchanged (the A/B control).
#
# The two cfg scales are build ARGs, not hardcoded, so a cfg sweep is `--build-arg
# MODALITY_CFG_SCALE=.. --build-arg REPORT_CFG_SCALE=..` per combination rather than a hand-edited
# Dockerfile per image -- and because they're declared this late, every layer above (apt, conda,
# the whole pip install) stays cache-hit across the whole sweep; only this ENV layer and the COPY
# layers below it rebuild per combination.
ARG MODALITY_CFG_SCALE=1.0
ARG REPORT_CFG_SCALE=1.0
ENV MRFLOW_MODALITY_CFG_SCALE=${MODALITY_CFG_SCALE} \
    MRFLOW_REPORT_CFG_SCALE=${REPORT_CFG_SCALE} \
    MRFLOW_NATIVE_SPACING_MODE=sample \
    MRFLOW_NATIVE_SPACING_TOPK=8 \
    MRFLOW_OUTPUT_DTYPE=float32

WORKDIR /opt/app
RUN mkdir -p /opt/app/models /input /output /checkpoint \
    && chown -R appuser:appuser /opt/app /input /output /checkpoint

COPY --chown=appuser:appuser echosyn/                   /opt/app/echosyn/
COPY --chown=appuser:appuser auto_regressive_generate/   /opt/app/auto_regressive_generate/
COPY --chown=appuser:appuser submission/predict.py            /opt/app/predict.py
COPY --chown=appuser:appuser submission/entrypoint.sh          /opt/app/entrypoint.sh
COPY --chown=appuser:appuser submission/native_spacing.py      /opt/app/native_spacing.py
COPY --chown=appuser:appuser submission/native_spacing_table.json /opt/app/native_spacing_table.json
COPY --chown=appuser:appuser submission/config.yaml              /opt/app/config.yaml
RUN chmod +x /opt/app/entrypoint.sh

USER appuser
ENTRYPOINT ["/opt/app/entrypoint.sh"]
