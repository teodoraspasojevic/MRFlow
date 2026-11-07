"""Debug launcher for the NeuroBERT-tokens arm's training: edit the block below, then
`python tools/debug_train_neurobert.py` (or run it under a debugger) from the repo root.

One process, one GPU, no wandb, no worker processes: breakpoints inside lvfm/train.py, the dataset
and the model all work. The real config is read, overridden only by the values below, written to
<OUTPUT_DIR>/debug_config.yaml, and lvfm/train.py runs on it unchanged.
"""
import os
import runpy
import sys

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------------- parameters
CONFIG = "lvfm/configs/mrflow_from_scratch_neurobert-tokens_ynorm_trim-v1.yaml"
OUTPUT_DIR = "/hnvme/workspace/y100dc19-mrflow-final/experiments/debug_neurobert"  # fresh dir: no resume
GPU = "0"
OVERRIDES = {
    "max_train_steps": 20,                  # globals.max_steps is only its default
    "gradient_accumulation_steps": 1,
    "dataloader.args.batch_size": 2,
    "dataloader.args.num_workers": 0,       # data loading in this process, so it is debuggable
    "dataloader.args.persistent_workers": False,
    "checkpointing_steps": 10,
    # train.py validates and checkpoints at step 0 and every N steps after. The block FID/FVD and
    # report-alignment metrics take ~20 min a fire on an A100, so they are off here; set them back
    # to the config's dicts (validation.fid_fvd / validation.alignment) to debug them.
    "validation.steps": 10,
    "validation.samples": 1,
    "validation.fid_fvd": None,
    "validation.alignment": None,
    "resume_from_checkpoint": None,
    "wandb_args.id": None,                  # a debug run never resumes the real run's wandb id
    # "mri.embedding_root": "...",          # e.g. a local copy of the embeddings
    # "mri.null_text": "...",
}
ENV = {
    # Fine-tuned NeuroBERT weights, if not at mrrate.NEUROBERT_WEIGHTS' default path.
    # "MRFLOW_NEUROBERT_WEIGHTS": "/path/to/neurobert-mlm-rsm/model.safetensors",
    "HF_HUB_OFFLINE": "1",
}
# ----------------------------------------------------------------------------------------------

os.environ.update(ENV)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)

config = OmegaConf.load(CONFIG)
for key, value in OVERRIDES.items():
    OmegaConf.update(config, key, value, force_add=True)
config.output_dir = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
debug_config = os.path.join(OUTPUT_DIR, "debug_config.yaml")
OmegaConf.save(config, debug_config)
print(f"debug config: {debug_config}")

sys.argv = ["lvfm/train.py", "--config", debug_config, "--no_wandb"]
runpy.run_path("lvfm/train.py", run_name="__main__")
