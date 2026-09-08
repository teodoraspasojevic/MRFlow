"""Post-processing helpers for `predict.py`, model-agnostic and self-contained.

Both modules here rewrite a *generated* volume's voxel grid onto a realistic MR-RATE one before it
is written as NIfTI -- `native_spacing` on the slice axis, `inplane_resample` on the other two --
each paired with the survey table it draws from. They import nothing from MRFlow itself (numpy and
scipy only), which is why they live in their own package: the rollout in `predict.py` is unaffected
by them, and either can be switched off at runtime (`MRFLOW_NATIVE_SPACING_MODE=off` /
`MRFLOW_INPLANE_MODE=off`) without touching the model path.

Mirrors `R2V-MR-Generation/submission/helpers/`, where the ports came from.
"""
