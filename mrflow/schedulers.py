# Adapted from CTFlow (https://github.com/WongJiayi/CTFlow); modified for MRFlow.
import math

from torch.optim.lr_scheduler import _LRScheduler


class StepBasedLearningRateScheduleWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        ref_steps=100_000,
        eta_min=0,
        decay_rate=1.0,
        last_epoch=-1,
    ):
        """
        Linear warm-up followed by inverse square root decay.

        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for the warm-up phase.
            ref_steps: Reference steps for decay adjustment.
            eta_min: Minimum learning rate.
            decay_rate: Decay rate multiplier.
            last_epoch: The index of last epoch.
        """
        self.warmup_steps = warmup_steps
        self.ref_steps = ref_steps
        self.decay_rate = decay_rate
        self.eta_min = eta_min
        super(StepBasedLearningRateScheduleWithWarmup, self).__init__(
            optimizer, last_epoch
        )
        self.config = {
            "warmup_steps": warmup_steps,
            "ref_steps": ref_steps,
            "eta_min": eta_min,
            "decay_rate": decay_rate,
        }

    def get_lr(self, step=None):
        current_step = step or self.last_epoch
        if current_step < self.warmup_steps:
            warmup_factor = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            decay_steps = current_step - self.warmup_steps
            lr = self.ref_lr / math.sqrt(
                1 + (decay_steps / self.ref_steps) * self.decay_rate
            )
            return [max(lr, self.eta_min) for _ in self.base_lrs]

    @property
    def ref_lr(self):
        return self.base_lrs[0]
