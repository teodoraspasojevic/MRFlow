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
#
# CUDA 12.1.1, not 12.8 -- confirmed the hard way (mrflow_submission_mod1-rep1's first submission,
# 2026-08-31): a cu128 image (nvidia/cuda:12.8.1-runtime-ubuntu22.04) produced exactly this
# entrypoint FATAL on the real platform, twice, while an R2V-MR-Generation submission running
# concurrently on the same platform -- same driver-detection mechanism, same immediate
# torch.cuda.is_available() check, no retry in either -- succeeded. The one confirmed difference
# between the two images' base layers: each nvidia/cuda tag declares (`NVIDIA_REQUIRE_CUDA` in the
# base image's own ENV) which host driver *bands* its CUDA runtime accepts, and they are discrete
# bands, not a simple floor -- 12.8.1-runtime lists driver 470.x, 535.x, 550.x, 560.x, 565.x;
# 12.1.1-cudnn8-runtime lists 470.x and 525.x. If the platform's nodes run driver 525.x (a common,
# stable cloud choice, and the version R2V's proven-working image explicitly targets), that band is
# simply absent from the 12.8 image's list -- nvidia-container-cli then declines to expose the GPU,
# which surfaces to PyTorch as a clean, silent `torch.cuda.is_available() == False`, independent of
# GPU count. Matching R2V's exact base tag (and, below, its exact torch/torchvision/cudnn pins) is
# the safest fix: it is the version already proven to negotiate this platform's driver.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bzip2 ca-certificates \
        gcc g++ \
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

# torch/torchvision/cudnn pins copied verbatim from R2V-MR-Generation/submission/dockerfiles/
# Dockerfile (its own comment there explains the --no-deps + explicit nvidia-cudnn-cu12==9.1.1.17:
# torch==2.5.1+cu121's wheel metadata pins nvidia-cudnn-cu12==9.1.0.70 exactly, and that patch
# build has been pulled from the index -- 9.1.1.17 is the same 9.1.x line, one patch up, otherwise
# byte-identical). This is the proven-on-this-platform combination (see the FROM comment above),
# so it is reused exactly rather than re-derived. xformers is NOT part of R2V's own stack (it uses
# monai, not xformers), so its pin below is new: 0.0.28.post3 is the xformers release built
# against torch 2.5.1, per its own published torch/xformers compatibility matrix, installed from
# the same cu121 index as torch so the CUDA build lines cannot mismatch.
RUN pip install --no-cache-dir --no-deps \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        filelock typing-extensions networkx jinja2 fsspec pillow setuptools \
        sympy==1.13.1 \
        triton==3.1.0 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.1.17 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.21.5 \
        nvidia-nvtx-cu12==12.1.105 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        xformers==0.0.28.post3 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        diffusers==0.31.0 \
        transformers==5.14.1 \
        tokenizers==0.22.2 \
        safetensors==0.8.0 \
        huggingface-hub==1.23.0 \
        accelerate==1.1.1 \
        wandb==0.18.5 \
        torchdiffeq==0.2.5 \
        einops==0.8.2 \
        timm==0.9.16 \
        omegaconf==2.3.1 \
        nibabel==5.4.2 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        scikit-image==0.26.0 \
        opencv-python-headless \
        imageio==2.37.4 \
        imageio-ffmpeg==0.6.0 \
        Pillow \
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
# MRFLOW_BATCH_SIZE: how many volumes predict.py's main loop batches into one shared
# generate_rollout() call (one Euler integration per block, covering every member at once) --
# see auto_regressive_generate.LatentAutoregressiveGenerator.generate_rollout and the batch-size
# sweep in submission/README.md for the measured throughput-vs-batch-size curve this value was
# picked from. A build ARG like the two cfg scales, so it can be swept via --build-arg without
# hand-editing the Dockerfile, and declared late enough that the whole pip-install layer above
# stays cache-hit across the sweep.
#
# MRFLOW_USE_BF16=1: run the denoiser in bfloat16 autocast rather than fp32 -- the same
# --mixed_precision bf16 regime training itself uses (see CLAUDE.md), so this is not a new
# numerical regime for the checkpoint, just skipped at inference until now.
#
# MRFLOW_USE_COMPILE=1: torch.compile(dynamic=True) on the denoiser via
# auto_regressive_generate.safe_compile, which suppresses dynamo errors and falls back to eager on
# any compile failure -- safe to enable unconditionally, worst case is no speedup.
ARG MODALITY_CFG_SCALE=1.0
ARG REPORT_CFG_SCALE=1.0
ARG BATCH_SIZE=2
ENV MRFLOW_MODALITY_CFG_SCALE=${MODALITY_CFG_SCALE} \
    MRFLOW_REPORT_CFG_SCALE=${REPORT_CFG_SCALE} \
    MRFLOW_BATCH_SIZE=${BATCH_SIZE} \
    MRFLOW_MAX_BLOCKS=16 \
    MRFLOW_NATIVE_SPACING_MODE=sample \
    MRFLOW_NATIVE_SPACING_TOPK=8 \
    MRFLOW_OUTPUT_DTYPE=float32 \
    MRFLOW_USE_BF16=1 \
    MRFLOW_USE_COMPILE=1

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
