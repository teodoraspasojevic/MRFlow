# GenerateCT

Cascaded text-to-CT: a CT-ViT tokenizer, a MaskGIT transformer that generates a 128x128x201 volume
from a text prompt, and a text-conditional diffusion super-resolution stage that takes it to
512x512x201. The oldest and most-cited of the three, and the standard comparison point in this
literature.

    upstream   https://github.com/ibrahimethemhamamci/GenerateCT  @ 2a81135 (2024-07-03)
    clone      $WS/baselines/upstream/GenerateCT
    weights    huggingface.co/generatect/GenerateCT/pretrained_models/{ctvit,transformer,superres}_pretrained.pt

## What has to happen before it is a baseline

**It is a chest-CT model and has to be fine-tuned on MR-RATE brain MRI**, the same as Text2CT. That
is the bulk of the work here, and it is worth deciding early what "fine-tuned" means: all three
stages, or the transformer only on a frozen CT-ViT. Whatever is chosen has to be stated next to the
number, because a reader will otherwise assume the former.

Three stages means three trainings and three checkpoints, and the README's own note that the
transformer needs an 80 GB A100 for inference maps onto one h200 per task here.

## Points where it will fight the contract

- **Fixed 201 slices.** The cascade generates exactly 201, where MR-RATE volumes run 143-200 at
  1 mm and MRFlow's rollout length is free. `canonicalize_external` will not stretch or cap it, so
  the mismatch stays visible in the per-plane slice counts, which is the right outcome — but it
  does mean this baseline cannot get the rollout-length question wrong the way MRFlow can, and the
  FVD/FID comparison is not length-matched. Say so when quoting.
- **Text encoder is T5.** Nothing to reconcile — a baseline uses its own encoder — but it is the
  most visible difference from MRFlow's CXR-BERT and belongs in the table's caption.
- **Anisotropic output.** 512x512x201 over a chest FOV; after MR fine-tuning the spacing is
  whatever the fine-tuning grid was. `ingest.py` reads it off the affine, so the only requirement
  is that the saver writes a *correct* affine — `inference_superres.py:187` builds one explicitly,
  so check what it puts there rather than assuming it is the training spacing.
- **`.txt`-scanning input.** `inference_transformer.py` collects prompts by globbing `*.txt`. The
  adapter should drive the model per case from `cases-test-n100.json` instead, so a prompt maps to
  a known `case_id` and the output filename is the one `ingest.py` expects.

## Adapter: what still has to be written

`slurms/generatect_run_shards.sh` + a driver that, for each case of `baselines/cases-test-n100.json`, builds the
prompt from that study's report with **GenerateCT's own** text formatting, runs the transformer and
then the super-resolution stage, and saves `<case_id>.nii.gz`. Its own venv in the workspace; no
MRFlow import on this side.
