"""Import the pinned CCELLA checkout, and refuse to run against one that is not the pinned state.

Upstream is a clone in the workspace, run as `python -m scripts.<module>` from its own root, so its
modules are a top-level `scripts` package. This puts that root on `sys.path` once and imports from
it -- no vendoring, no copy, and `scripts.utils.define_instance` keeps resolving `_target_` blocks
exactly as it does upstream.

**One upstream file is edited**, `scripts/diff_model_train_all.py::train_one_epoch`, 21 insertions
and 3 deletions against commit `619dfbb6`. `upstream.patch` next to this file is that diff.
`verify_upstream` checks both the commit and that the working tree is exactly the pinned commit
plus that patch -- it reverses the patch in memory and compares, so a drifted checkout, a partially
applied patch, or an upstream update all fail loudly instead of training something else.

Why the edit could not be avoided: `train_one_epoch` runs a whole epoch and returns. Step-based
checkpointing at an explicit step list, validation at a step interval, per-step W&B logging and an
exact-step resume all need control *inside* that loop, and no amount of dependency injection
reaches inside a function that does not yield. The patch adds a `step_hook` callback and nothing
else that changes arithmetic; the four other hunks widen a hard-coded class count to the loss's own
width, pass the modality id through, name the diffusion term before the classification term is
added to it, and keep the gradient norm the optimizer already computed. Every line that samples
noise, calls the U-Net, forms the loss, steps the scaler and steps the scheduler is untouched.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_PATH = os.path.join(HERE, "upstream.patch")
PINNED_COMMIT = "619dfbb6f5c20eaa6724a970485eb0a95c22146f"
DEFAULT_ROOT = "/hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/CCELLA"
PATCHED_FILES = ("scripts/diff_model_train_all.py",)


class UpstreamError(RuntimeError):
    """The upstream checkout is missing, at the wrong commit, or not in the pinned patched state."""


def upstream_root():
    """`MRFLOW_CCELLA_ROOT` if set, else the workspace clone."""
    return os.environ.get("MRFLOW_CCELLA_ROOT", DEFAULT_ROOT)


_ON_PATH = set()


def upstream_module(name):
    """Import `scripts.<x>` from the pinned checkout. `name` is e.g. `"scripts.text_ldm"`."""
    root = upstream_root()
    if not os.path.isdir(os.path.join(root, "scripts")):
        raise UpstreamError(f"no CCELLA checkout at {root} (set MRFLOW_CCELLA_ROOT)")
    if root not in _ON_PATH:
        sys.path.insert(0, root)
        _ON_PATH.add(root)
    return importlib.import_module(name)


def _git(root, *args):
    result = subprocess.run(("git", "-C", root) + args, capture_output=True, text=True)
    if result.returncode:
        raise UpstreamError(f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}")
    return result.stdout


def verify_upstream(root=None, strict=True):
    """`(ok, [problems])` for the checkout: right commit, right patch, nothing else changed.

    With `strict=False` the commit check is reported but not required, which is what a developer
    re-pinning upstream wants; training always calls it strict.
    """
    root = root or upstream_root()
    problems = []

    if not os.path.isdir(os.path.join(root, ".git")):
        return False, [f"{root} is not a git checkout"]

    head = _git(root, "rev-parse", "HEAD").strip()
    if head != PINNED_COMMIT:
        message = f"upstream HEAD is {head[:12]}, pinned is {PINNED_COMMIT[:12]}"
        problems.append(message) if strict else problems.append("(non-strict) " + message)

    # Exactly the patched files may differ, and only by the stored patch. Byte-compiled caches are
    # created by the act of importing upstream, so they are not evidence of a modified checkout.
    changed = set()
    for line in _git(root, "status", "--porcelain").splitlines():
        path = line[3:].strip()
        if "__pycache__" in path or path.endswith(".pyc"):
            continue
        changed.add(path)
    unexpected = changed - set(PATCHED_FILES)
    if unexpected:
        problems.append(f"unexpected local changes in the pinned checkout: {sorted(unexpected)}")

    with open(PATCH_PATH, "rb") as handle:
        patch = handle.read()
    applied = subprocess.run(("git", "-C", root, "apply", "--check", "--reverse", "-"),
                             input=patch, capture_output=True)
    if applied.returncode:
        appliable = subprocess.run(("git", "-C", root, "apply", "--check", "-"),
                                   input=patch, capture_output=True)
        if appliable.returncode == 0:
            problems.append(
                f"upstream.patch is NOT applied to {root}. Apply it with:\n"
                f"    git -C {root} apply {PATCH_PATH}")
        else:
            problems.append(
                f"upstream.patch neither applies nor reverse-applies to {root}: the checkout has "
                f"drifted from commit {PINNED_COMMIT[:12]} or the patch is stale.\n"
                f"    {appliable.stderr.decode().strip()}")
    return not problems, problems


def require_upstream(root=None):
    """Raise unless the checkout is exactly the pinned commit plus `upstream.patch`."""
    ok, problems = verify_upstream(root)
    if not ok:
        raise UpstreamError("upstream CCELLA checkout is not in the pinned state:\n  - "
                            + "\n  - ".join(problems))
