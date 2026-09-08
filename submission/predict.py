#!/usr/bin/env python3
"""VLM3D `mr-volume-generation` entry point: `/input/prompts.json` -> `/output/*.nii.gz`.

MRFlow's own STDiT + flow-matching autoregressive generator (`auto_regressive_generate`), reused
unchanged -- this is the same rollout `evaluation/main.py` scores. What's new here is the platform
I/O contract, decoding (modality, plane) out of each `input_image_name`, turning the challenge's one
flat report string into the {findings, impression} sections `encode_conditioning` was trained
against, writing a NIfTI whose affine matches the axis order the model rolls out in, multi-GPU
launch and checkpoint/resume.

Fail-loud: no try/except around the main loop. A missing output scores as invalid, which is worse
than crashing loudly (mirrors R2V-MR-Generation's `submission/predict/predict.py`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
HELPERS_DIR = _HERE / "helpers"  # the grid-rewriting helpers and their survey tables
sys.path.insert(0, str(_HERE))  # so `helpers.*` imports below resolve next to this file

# ---- container layout (FORITHMUS_* env vars are authoritative when the platform sets them) ----
INPUT_DIR = Path(os.environ.get("FORITHMUS_INPUT", "/input"))
OUTPUT_DIR = Path(os.environ.get("FORITHMUS_OUTPUT", "/output"))
CHECKPOINT_DIR = Path(os.environ.get("FORITHMUS_CHECKPOINT", "/checkpoint"))
WEIGHTS_DIR = Path(os.environ.get("FORITHMUS_WEIGHTS", "/weights"))
MODELS_DIR = Path(os.environ.get("MRFLOW_MODELS_DIR", "/opt/app/models"))

VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".npy", ".npz")


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


# ─────────────────────────── multi-GPU (torchrun) ───────────────────────────

def ddp_setup() -> tuple[int, int, int, bool]:
    """`(rank, world_size, local_rank, is_ddp)`. Inference-only process-group coordination
    (rank/world_size + a final barrier), not gradient-synchronizing DDP -- each rank loads its own
    full model and generates a disjoint slice of the prompt list. Only active under `torchrun`
    (which sets `RANK`/`WORLD_SIZE`); plain `python predict.py` returns `(0, 1, 0, False)`.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise SystemExit("RANK/WORLD_SIZE are set (torchrun launch) but no GPU is visible")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def ddp_cleanup(is_ddp: bool) -> None:
    if not is_ddp:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────── case id -> modality, plane ───────────────────────────
# `input_image_name` is `{study_uid}_{modality}-raw-{plane}`, optionally with a `-{n}` duplicate-
# series suffix (e.g. `WNPYIQCPIN_t1w-raw-sag-2`). `mrrate.py`'s own alias tables are the single
# source of truth for what a code resolves to -- reused here rather than a second copy, so a code
# means the same thing in the class-id path and in the acquisition-prefix text.
NAME_PATTERN = re.compile(
    r"[-_](?P<modality>[a-z0-9]+)-raw-(?P<plane>[a-z]+)(?:-\d+)?$", re.IGNORECASE
)


def modality_plane_for(case_id: str) -> tuple[str, str]:
    """Decode `(modality, plane)` canonical names from a case id's own
    `..._<modality>-raw-<plane>` suffix, via `mrrate.py`'s MODALITY_ALIASES/PLANE_ALIASES --
    e.g. `t1w`->`T1w`, `swi`->`SWI`, `axi`->`AXIAL`, `obl`->`AXIAL` (no oblique class; see
    `mrrate.PLANE_ALIASES`, which treats an oblique acquisition as a tilted axial)."""
    from echosyn.common.mrrate import MODALITY_ALIASES, MODALITY_TO_ID, PLANE_ALIASES, PLANE_TO_ID

    match = NAME_PATTERN.search(case_id)
    if not match:
        raise ValueError(f"case id {case_id!r} does not encode '..._<modality>-raw-<plane>'")
    modality_code, plane_code = match.group("modality").lower(), match.group("plane").lower()
    modality = MODALITY_ALIASES.get(modality_code, modality_code)
    plane = PLANE_ALIASES.get(plane_code, PLANE_ALIASES.get(plane_code.upper(), plane_code))
    if modality not in MODALITY_TO_ID or plane not in PLANE_TO_ID:
        raise ValueError(f"case id {case_id!r} names an unrecognised modality/plane code "
                         f"({modality_code!r}, {plane_code!r})")
    return modality, plane


# ─────────────────────────── prompt parsing ───────────────────────────

def case_stem(name: str) -> str:
    """Strip a known volume extension; the output file is `<stem>.nii.gz`."""
    text = str(name).strip()
    lowered = text.lower()
    for suffix in VOLUME_SUFFIXES:
        if lowered.endswith(suffix):
            return text[: -len(suffix)]
    return text


def read_prompts(input_dir: Path) -> list[tuple[str, str]]:
    """`[(case_id, report_text)]` from the prompt file under `input_dir`."""
    candidates = sorted(input_dir.rglob("*.json"))
    if not candidates:
        raise SystemExit(f"no JSON prompt file under {input_dir}")
    exact = [p for p in candidates if p.name == "prompts.json"]
    promptish = [p for p in candidates if "prompt" in p.name.lower()]
    other = [p for p in candidates if p.name.lower() != "metadata.json"]
    path = (exact or promptish or other or candidates)[0]
    print(f"reading prompts from {path}")

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("prompts", "cases", "data", "inputs"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not a JSON array (or an object wrapping one)")

    id_keys = ("input_image_name", "case_id", "id", "name", "output_image_name")
    text_keys = ("report", "text", "prompt", "findings", "report_text")
    prompts: list[tuple[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SystemExit(f"{path} entry {index} is {type(item).__name__}, expected an object")
        raw_id = next((item[k] for k in id_keys if item.get(k)), None)
        if raw_id is None:
            raise SystemExit(f"{path} entry {index} has no case-id field (tried {id_keys})")
        report = next((item[k] for k in text_keys if item.get(k)), "")
        prompts.append((case_stem(raw_id), str(report)))

    stems = [s for s, _ in prompts]
    duplicates = {s for s in stems if stems.count(s) > 1}
    if duplicates:
        raise SystemExit(f"duplicate case ids in {path}: {sorted(duplicates)[:5]}")

    print(f"{len(prompts)} prompt(s)")
    return prompts


# Section headings MR-RATE's own structuring step emits -- see `echosyn.common.mrrate.format_report`,
# which is what `encode_conditioning` was trained against (`sections=("findings", "impression")`).
_SECTION_HEADINGS = {
    "clinical_information": r"clinical(?:\s+information|\s+history)?|history|indication",
    "technique": r"technique|protocol",
    "findings": r"findings?|description",
    "impression": r"impression|conclusion|summary|assessment",
}


def split_sections(report: str) -> dict[str, str]:
    """One flat report string -> `{section: text}`, matching the {findings, impression} dict shape
    `read_report`/MR-RATE's own `report.json` has at training time. The challenge hands us the
    study's original free-text report (headings and all: "Findings:", "Impression:", ...), so
    splitting it back out here recovers the same sections training encoded."""
    text = str(report or "").strip()
    if not text:
        return {}
    pattern = "|".join(f"(?P<{name}>{alts})" for name, alts in _SECTION_HEADINGS.items())
    matches = list(re.finditer(rf"(?<![A-Za-z])(?:{pattern})\s*:", text, re.IGNORECASE))
    if not matches:
        return {"findings": text}
    sections: dict[str, str] = {}
    for position, match in enumerate(matches):
        name = match.lastgroup
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections[name] = f"{sections[name]} {body}".strip() if name in sections else body
    # Preamble before the first heading (e.g. "44-year-old female:") is clinical context.
    preamble = text[: matches[0].start()].strip().rstrip(":").strip()
    if preamble:
        existing = sections.get("clinical_information", "")
        sections["clinical_information"] = f"{preamble} {existing}".strip()
    return sections or {"findings": text}


def seed_for_case(case_id: str, base_seed: int) -> int:
    """A per-case seed, deterministic in `case_id` alone -- not a call counter, a DDP rank, or
    `hash()` (Python salts that per process) -- so a resumed or multi-rank run redraws the same
    noise for the same case, matching `evaluation/main.py`'s own "seeded off the case" convention."""
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()[:8]
    return (base_seed + int.from_bytes(digest, "big")) % (2 ** 31)


# ─────────────────────────── checkpoint / resume ───────────────────────────
# The platform can preempt or time out a long batch job (a full-body rollout is ~20 blocks of 201
# Euler steps per case), so finished volumes are backed up to /checkpoint (survives a restart,
# unlike /output) and re-copied back on resume. `done.json` is rank-scoped under multi-GPU.

def checkpoint_paths(rank: int, world_size: int) -> tuple[Path, Path]:
    suffix = f"_rank{rank}" if world_size > 1 else ""
    return CHECKPOINT_DIR / f"done{suffix}.json", CHECKPOINT_DIR / "outputs"


def load_done(done_file: Path, backup_dir: Path) -> list[str]:
    """Finished output filenames from a prior run, verified to still exist somewhere."""
    if not done_file.exists():
        return []
    done = []
    for filename in json.loads(done_file.read_text()):
        backup, output = backup_dir / filename, OUTPUT_DIR / filename
        if output.exists():
            done.append(filename)
        elif backup.exists():
            shutil.copy2(backup, output)
            done.append(filename)
    return done


def save_done(done_file: Path, done: list[str]) -> None:
    done_file.write_text(json.dumps(done))


def mark_done(filename: str, done: list[str], done_file: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_DIR / filename, backup_dir / filename)
    done.append(filename)
    save_done(done_file, done)


def install_shutdown_handler(done: list[str], done_file: Path, rank: int) -> None:
    """SIGTERM gives 30s before SIGKILL -- just persist the index, not a full volume copy."""
    def handle(signum, _frame):
        print(f"[rank {rank}] signal {signum} received after {len(done)} case(s); saving and exiting")
        save_done(done_file, done)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


# ─────────────────────────── model / weight resolution ───────────────────────────

def resolve_dir(name_env: str, default_name: str) -> Path:
    """A model directory, looked up in `/opt/app/models` (symlinked by entrypoint.sh from
    `/weights`) then `/weights` directly."""
    name = env_str(name_env, default_name)
    candidates = [Path(name)] if Path(name).is_absolute() else [MODELS_DIR / name, WEIGHTS_DIR / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"{name_env}={name!r} not found. Looked in: {[str(c) for c in candidates]}")


def resolve_baked(name_env: str, default_name: str) -> Path:
    """A small file baked into the image -- config.yaml next to this script, the two spacing
    tables in helpers/ -- overridable through /weights or /opt/app/models the same way a checkpoint
    is. Both directories are searched under the bare name, so the env overrides take the same value
    they did when the tables sat next to predict.py."""
    name = env_str(name_env, default_name)
    candidates = ([Path(name)] if Path(name).is_absolute()
                  else [_HERE / name, HELPERS_DIR / name, MODELS_DIR / name, WEIGHTS_DIR / name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"{name_env}={name!r} not found. Looked in: {[str(c) for c in candidates]}")


def build_generator(config, device: str):
    """`(generator, tokenizer, text_encoder)`. Mirrors `evaluation/main.py::build_generator`, plus
    the text encoder MRFlow's own `preprocess_mrrate.py` builds separately."""
    import torch

    from auto_regressive_generate import LatentAutoregressiveGenerator
    from echosyn.common import get_vae_scaler, instantiate, instantiate_class_from_config
    from echosyn.common.mrrate import build_text_encoder

    denoiser_dir = resolve_dir("MRFLOW_DENOISER_DIR", "denoiser_ema")
    vae_dir = resolve_dir("MRFLOW_VAE_DIR", "flux-vae-f8-16ch")
    text_dir = resolve_dir("MRFLOW_TEXT_ENCODER_DIR", "BiomedVLP-CXR-BERT-specialized")
    # The saved config carries the Helma cluster's own paths; only these two are ever read again
    # (get_vae_scaler reads vae.pretrained/config.json; text_checkpoint is otherwise informational).
    config.vae.pretrained = str(vae_dir)
    config.mri.text_checkpoint = str(text_dir)

    print(f"denoiser={denoiser_dir} vae={vae_dir} text_encoder={text_dir} device={device}")
    denoiser = instantiate_class_from_config(config.denoiser)
    denoiser = denoiser.from_pretrained(str(denoiser_dir)).to(device).eval()
    vae = instantiate(config.vae).eval().to(device)
    tokenizer, text_encoder = build_text_encoder(str(text_dir), device)

    generator = LatentAutoregressiveGenerator(
        denoiser=denoiser, vae=vae, device=device,
        vae_scaling=get_vae_scaler(config, device), config=config,
        block_size=config.globals.target_nframes,
        modality_cfg_scale=env_float("MRFLOW_MODALITY_CFG_SCALE", config.guidance.modality_cfg_scale),
        report_cfg_scale=env_float("MRFLOW_REPORT_CFG_SCALE", config.guidance.report_cfg_scale),
        use_bf16=env_str("MRFLOW_USE_BF16", "1") == "1",
        use_compile=env_str("MRFLOW_USE_COMPILE", "1") == "1",
    )
    return generator, tokenizer, text_encoder


# ─────────────────────────── writing the NIfTI ───────────────────────────
# The generated array is always `(T, H, W)` with the slice axis leading (axis 0) -- what
# `plane_order` in echosyn/common/mrrate.py permutes an RAS-canonical volume into before the model
# ever sees it, and what `LatentAutoregressiveGenerator.decode_latent` returns unchanged. Which
# physical RAS direction each array axis is depends only on the plane:
#
#   AXIAL:    axis0 -> S(uperior), axis1 -> R(ight),    axis2 -> A(nterior)   (SRA)
#   SAGITTAL: axis0 -> R(ight),    axis1 -> S(uperior),  axis2 -> A(nterior)  (RSA)
#   CORONAL:  axis0 -> A(nterior), axis1 -> S(uperior),  axis2 -> R(ight)     (ASR)
#
# (`plane_order`'s (S, R, A) index tuples, named out.) No axis is ever flipped -- `read_canonical`
# only permutes RAS-canonical data, never mirrors it -- so this is a signed permutation of RAS with
# every sign positive: an honestly-labeled, if non-identity, affine.
_PLANE_DIRECTION = {
    "AXIAL": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "SAGITTAL": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "CORONAL": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
}


def build_affine(plane: str, spacing_by_axis: tuple[float, float, float], shape: tuple[int, ...]):
    """4x4 NIfTI affine for a `(T, H, W)` array in the axis order `plane` implies, mapping voxel
    indices to RAS mm with the volume centered on the origin. `spacing_by_axis` is per array axis:
    axis0 may differ from the model's native 1 mm after native-spacing slab-averaging, and axes 1/2
    after in-plane spline upsampling."""
    directions = _PLANE_DIRECTION[plane]
    affine = np.eye(4)
    for axis in range(3):
        affine[:3, axis] = np.array(directions[axis], dtype=np.float64) * spacing_by_axis[axis]
    center_index = (np.array(shape[:3], dtype=np.float64) - 1.0) / 2.0
    affine[:3, 3] = -affine[:3, :3] @ center_index
    return affine


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    started = time.time()
    rank, world_size, local_rank, is_ddp = ddp_setup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    import torch
    from omegaconf import OmegaConf

    from echosyn.common import check_label_mapping
    from echosyn.common.mrrate import encode_conditioning, modality_to_id, plane_to_id
    from helpers.inplane_resample import InplaneSpacingTable, to_inplane_grid
    from helpers.native_spacing import (DEFAULT_MODE, DEFAULT_TOP_K, MODES, NativeSpacingTable,
                                        to_native_grid)

    prompts = read_prompts(INPUT_DIR)  # same file, same parse, identical on every rank
    done_file, backup_dir = checkpoint_paths(rank, world_size)
    done = load_done(done_file, backup_dir)
    install_shutdown_handler(done, done_file, rank)
    if done:
        print(f"[rank {rank}] resuming: {len(done)} case(s) already done")

    config_path = resolve_baked("MRFLOW_CONFIG", "config.yaml")
    config = OmegaConf.load(config_path)
    check_label_mapping(config)

    native_mode = env_str("MRFLOW_NATIVE_SPACING_MODE", DEFAULT_MODE)
    if native_mode not in MODES:
        raise SystemExit(f"MRFLOW_NATIVE_SPACING_MODE={native_mode!r} must be one of {MODES}")
    native_table = None
    if native_mode != "off":
        table_path = resolve_baked("MRFLOW_NATIVE_SPACING_TABLE", "native_spacing_table.json")
        native_table = NativeSpacingTable.load(
            table_path, top_k=env_int("MRFLOW_NATIVE_SPACING_TOPK", DEFAULT_TOP_K), mode=native_mode)
        print(f"[rank {rank}] native-spacing table={table_path} mode={native_mode} "
              f"top_k={native_table.top_k}")
    else:
        print(f"[rank {rank}] native-spacing mode=off -- writing the generated 1mm grid unchanged")

    # A separate axis from the slice-axis rewrite above, with its own on/off switch rather than
    # being bundled into MRFLOW_NATIVE_SPACING_MODE -- see inplane_resample.py. Both tables are
    # loaded here, before the model, so a missing or malformed one fails in the first seconds
    # rather than after the first volumes are already written.
    inplane_mode = env_str("MRFLOW_INPLANE_MODE", DEFAULT_MODE)
    if inplane_mode not in MODES:
        raise SystemExit(f"MRFLOW_INPLANE_MODE={inplane_mode!r} must be one of {MODES}")
    inplane_table = None
    if inplane_mode != "off":
        inplane_table_path = resolve_baked("MRFLOW_INPLANE_SPACING_TABLE",
                                           "inplane_spacing_table.json")
        inplane_table = InplaneSpacingTable.load(
            inplane_table_path, top_k=env_int("MRFLOW_INPLANE_TOPK", DEFAULT_TOP_K),
            mode=inplane_mode)
        print(f"[rank {rank}] inplane-spacing table={inplane_table_path} mode={inplane_mode} "
              f"top_k={inplane_table.top_k}")
    else:
        print(f"[rank {rank}] inplane mode=off -- leaving the generated 256^2 1mm grid unchanged")

    max_blocks = env_int("MRFLOW_MAX_BLOCKS",
                         config.mri.preprocess.max_slices // config.globals.target_nframes)
    base_seed = env_int("MRFLOW_SEED", config.seed)
    dtype = np.dtype(env_str("MRFLOW_OUTPUT_DTYPE", "float32"))

    device = f"cuda:{local_rank}" if is_ddp else env_str("MRFLOW_DEVICE", "cuda")
    generator, tokenizer, text_encoder = build_generator(config, device)
    print(f"[rank {rank}] modality_cfg_scale={generator.modality_cfg_scale} "
          f"report_cfg_scale={generator.report_cfg_scale} max_blocks={max_blocks}")

    # Striped, disjoint slice of the (1-indexed) prompt list -- rank 0 gets prompts 1, 1+N, ... --
    # one case per rollout. Batching several volumes into one Euler integration was tried and
    # removed: it is faster per volume, but noise can then only be seeded per batch rather than
    # per case, so a resumed run with a different grouping redraws different volumes.
    my_prompts = list(enumerate(prompts, start=1))[rank::world_size]
    for index, (case_id, report) in my_prompts:
        filename = f"{case_id}.nii.gz"
        if filename in done:
            continue

        modality, plane = modality_plane_for(case_id)
        sections = split_sections(report)

        torch.manual_seed(seed_for_case(case_id, base_seed))
        embedding = encode_conditioning(tokenizer, text_encoder, sections, modality, plane,
                                        max_length=config.mri.text_max_length)
        embedding = (embedding / (embedding.norm(p=2) + 1e-6)).unsqueeze(0).to(device)
        modality_id = torch.tensor([modality_to_id(modality)], device=device)
        plane_id = torch.tensor([plane_to_id(plane)], device=device)

        case_started = time.time()
        latent = generator.generate(embedding, modality_id, plane_id, max_blocks=max_blocks)
        if latent.shape[2] == 0:
            raise RuntimeError(f"{case_id}: every generated slice was a stop frame")
        volume = generator.decode_latent(latent)[0, 0].numpy().astype(np.float32)  # (T, H, W)

        spacing_by_axis = (1.0, 1.0, 1.0)
        note = f"grid={volume.shape} slice_spacing=1.00mm"
        if native_table is not None:
            target = native_table.draw(modality, plane, case_id)
            volume, slice_spacing, info = to_native_grid(volume, target["thickness_mm"])
            spacing_by_axis = (slice_spacing, 1.0, 1.0)
            note = (f"grid {info['source_slices']}->{info['target_slices']} "
                    f"drew {target['thickness_mm']:.1f}mm got {info['slice_spacing_mm']:.2f}mm "
                    f"bucket={target['bucket']}")

        # Independent of the slice axis above, and applied after it so it upsamples whichever
        # slice grid this volume ended up on. Same FOV-preserving contract, same per-case draw.
        if inplane_table is not None:
            inplane_target = inplane_table.draw(modality, plane, case_id)
            volume, inplane_spacing, inplane_info = to_inplane_grid(
                volume, inplane_target["inplane_mm"])
            spacing_by_axis = (spacing_by_axis[0], *inplane_spacing)
            note += (f" | inplane {inplane_info['source_inplane']}->"
                     f"{inplane_info['target_inplane']} drew "
                     f"{inplane_target['inplane_mm']:.2f}mm got "
                     f"{inplane_info['inplane_spacing_mm'][0]:.2f}mm "
                     f"bucket={inplane_target['bucket']}")

        affine = build_affine(plane, spacing_by_axis, volume.shape)
        nib.save(nib.Nifti1Image(volume.astype(dtype, copy=False), affine), OUTPUT_DIR / filename)
        mark_done(filename, done, done_file, backup_dir)

        elapsed = time.time() - case_started
        print(f"[rank {rank}][{index}/{len(prompts)}] {case_id} {modality} {plane} "
              f"{note} {elapsed:.1f}s")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier(device_ids=[local_rank])

    exit_code = 0
    if rank == 0:
        written = {p.name for p in OUTPUT_DIR.glob("*.nii.gz")}
        print(f"wrote {len(written)}/{len(prompts)} volume(s) to {OUTPUT_DIR} "
              f"in {(time.time() - started) / 60:.1f} min")
        missing = [f"{cid}.nii.gz" for cid, _ in prompts if f"{cid}.nii.gz" not in written]
        if missing:
            print(f"ERROR: {len(missing)} prompt(s) produced no volume, e.g. {missing[:5]}")
            exit_code = 1

    ddp_cleanup(is_ddp)
    if exit_code:
        raise SystemExit(exit_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
