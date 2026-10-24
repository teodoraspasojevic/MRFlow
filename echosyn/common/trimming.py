"""Blank-slice trimming: per-slice statistics and the policy that turns them into slice bounds.

A slice is *blank* when fewer than `min_foreground` of its pixels differ from the background by more
than `tau`. MR volumes reach the VAE normalized to [0, 1] over their nonzero voxels, with background
and padding exactly 0, so `tau` reads as a fraction of the volume's 99.5th-percentile intensity.

Only contiguous blank runs at the two ends of the slice axis are removed. Interior blank slices are
kept, in-plane extent is never touched, and slice order is preserved: the result is always one
contiguous `[start, stop)` window of the original axis 0.

Statistics are computed once, on a fixed grid of thresholds (`TAUS`), so a policy can be changed or
swept without re-reading a single volume.
"""

from dataclasses import asdict, dataclass

import numpy as np

# Intensity thresholds the scan records foreground fractions at. A policy's `tau` must be one of them.
TAUS = (0.01, 0.02, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class TrimPolicy:
    tau: float = 0.05             # |x - background| above this is foreground
    min_foreground: float = 0.01  # a slice with less foreground than this fraction is blank
    margin: int = 0               # blank slices kept on each side of the foreground
    min_slices: int = 32          # never trim below this; 2 * block_size for MRFlow
    all_blank: str = "exclude"    # "exclude" drops an all-blank volume, "keep" leaves it untrimmed
    background: float = 0.0

    def __post_init__(self):
        if self.tau not in TAUS:
            raise ValueError(f"tau={self.tau} is not a scanned threshold; choose from {TAUS}")
        if self.all_blank not in ("exclude", "keep"):
            raise ValueError(f"all_blank must be 'exclude' or 'keep', not {self.all_blank!r}")
        if self.margin < 0 or self.min_slices < 1:
            raise ValueError("margin must be >= 0 and min_slices >= 1")

    def to_dict(self):
        return asdict(self)


def slice_stats(volume, background=0.0):
    """(T, H, W) array -> dict of per-slice statistics, each a float32 array of length T.

    Non-finite pixels are counted, then treated as background, so a NaN never reads as foreground.
    """
    volume = np.asarray(volume, dtype=np.float32)
    flat = volume.reshape(volume.shape[0], -1)
    finite = np.isfinite(flat)
    dev = np.abs(np.where(finite, flat, background) - background)
    stats = {f"fg_{tau}": (dev > tau).mean(1).astype(np.float32) for tau in TAUS}
    clean = np.where(finite, flat, background)
    stats.update(
        mean=clean.mean(1).astype(np.float32),
        std=clean.std(1).astype(np.float32),
        p99=np.percentile(clean, 99, axis=1).astype(np.float32),
        nonfinite=(~finite).sum(1).astype(np.float32),
    )
    return stats


def blank_slices(stats, policy):
    """Boolean blank mask over the slice axis, from `slice_stats` output."""
    return np.asarray(stats[f"fg_{policy.tau}"]) < policy.min_foreground


def blank_runs(blank):
    """(leading, trailing, interior) blank-slice counts. An all-blank volume is all leading."""
    blank = np.asarray(blank, dtype=bool)
    T = len(blank)
    if blank.all():
        return T, 0, 0
    lead = int(np.argmax(~blank))
    trail = int(np.argmax(~blank[::-1]))
    return lead, trail, int(blank[lead:T - trail].sum())


def trim_bounds(blank, policy):
    """Blank mask -> (start, stop, reason). `start is None` means the volume is excluded.

    reason is one of:
        unchanged      no boundary blank slices (or only ones inside the margin)
        trimmed        boundary blank runs removed
        min_slices     trimming stopped early to keep `min_slices`, widened around the foreground
        short_source   the source is already below `min_slices`; left untrimmed
        all_blank      nothing is foreground; excluded or kept per `policy.all_blank`
    """
    T = len(blank)
    lead, trail, _ = blank_runs(blank)
    if lead == T:
        return (None, None, "all_blank") if policy.all_blank == "exclude" else (0, T, "all_blank")
    if T < policy.min_slices:
        return 0, T, "short_source"

    start = max(0, lead - policy.margin)
    stop = min(T, T - trail + policy.margin)
    if stop - start < policy.min_slices:
        # Grow symmetrically around the foreground, then shift the window back inside [0, T).
        grow = policy.min_slices - (stop - start)
        start -= grow // 2
        stop += grow - grow // 2
        if start < 0:
            stop, start = stop - start, 0
        if stop > T:
            start, stop = start - (stop - T), T
        return start, stop, "min_slices"
    return start, stop, ("unchanged" if (start, stop) == (0, T) else "trimmed")
