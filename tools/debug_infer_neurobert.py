"""Debug launcher for NeuroBERT-tokens inference: edit the block below, then
`python tools/debug_infer_neurobert.py` (or run it under a debugger) from the repo root.

It pulls one study's cached NeuroBERT artifact out of the embedding zip bundles into
<OUTPUT>/embedding.pt and runs auto_regressive_generate/main.py on it unchanged, in this process,
so breakpoints in the generator work. MODALITY / PLANE must be the series' own: a pooled report
embedding cannot be read back, so nothing checks they agree with the report. A full-body rollout
is up to 20 blocks of 201 Euler steps, ~47 s a block on an A100; an early checkpoint never emits
the stop token, so it runs all 20.
"""
import csv
import glob
import io
import os
import runpy
import sys
import zipfile

import torch

# ---------------------------------------------------------------------------------- parameters
EXPERIMENT = "/hnvme/workspace/y100dc19-mrflow-final/experiments/mrflow_from_scratch_neurobert-tokens_ynorm_trim-v1"
CONFIG = f"{EXPERIMENT}/config.yaml"           # the config saved into the experiment dir
CKPT = f"{EXPERIMENT}/checkpoint-100000/denoiser_ema"
EMBEDDING_ROOT = "/hnvme/workspace/y100dc19-mrflow-final/data/mrflow_neurobert-mlm-rsm_tokens"
SPLIT = "val"
STUDY_UID = None                                # None: the first study of SPLIT's first manifest
MODALITY = "T1w"                                # T1w | T2w | FLAIR | SWI | MRA
PLANE = "AXIAL"                                 # AXIAL | SAGITTAL | CORONAL
TYPE = "full-body"                              # full-body | gt-head | block-wise
GT_LATENT = None                                # needed for gt-head / block-wise
MODALITY_CFG_SCALE = None                       # None: config.guidance
REPORT_CFG_SCALE = None
OUTPUT = "debug_infer_out"
GPU = "0"
ENV = {
    # "MRFLOW_NEUROBERT_WEIGHTS": "/path/to/neurobert-mlm-rsm/model.safetensors",
    "HF_HUB_OFFLINE": "1",
}
# ----------------------------------------------------------------------------------------------

os.environ.update(ENV)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)


def find_artifact(study_uid):
    """(zip, member) of a study's artifact, from the embedding manifests."""
    for manifest in sorted(glob.glob(os.path.join(EMBEDDING_ROOT, "manifest", f"{SPLIT}-*.csv"))):
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if study_uid is None or row["embedding_path"] == f"embeddings/{study_uid}.pt":
                    return row["zip"], row["embedding_path"]
    raise FileNotFoundError(f"no {SPLIT} artifact for study {study_uid} under {EMBEDDING_ROOT}")


bundle, member = find_artifact(STUDY_UID)
with zipfile.ZipFile(os.path.join(EMBEDDING_ROOT, bundle)) as z:
    artifact = torch.load(io.BytesIO(z.read(member)), map_location="cpu")
os.makedirs(OUTPUT, exist_ok=True)
embedding = os.path.join(OUTPUT, "embedding.pt")
torch.save(artifact, embedding)
print(f"{member} from {bundle}: tag {artifact['neurobert']['tag']}, "
      f"{artifact['neurobert']['n_tokens']} tokens -> {embedding}")

sys.argv = ["auto_regressive_generate/main.py", "--config", CONFIG, "--ckpt", CKPT,
            "--embedding", embedding, "--output", OUTPUT, "--modality", MODALITY,
            "--plane", PLANE, "--type", TYPE]
if GT_LATENT:
    sys.argv += ["--gt-latent", GT_LATENT]
if MODALITY_CFG_SCALE is not None:
    sys.argv += ["--modality-cfg-scale", str(MODALITY_CFG_SCALE)]
if REPORT_CFG_SCALE is not None:
    sys.argv += ["--report-cfg-scale", str(REPORT_CFG_SCALE)]
runpy.run_path("auto_regressive_generate/main.py", run_name="__main__")
