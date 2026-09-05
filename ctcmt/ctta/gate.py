"""Score-EMA gate.

Skips an adaptation step when the teacher confidence has stabilized around
its EMA. Matches ``CTCMT_MTL._teacher_pseudo`` in the reference detectron2
adapter, and the earlier AMROD gate it was derived from.
"""
from __future__ import annotations

import torch


class ScoreEMGate:
    """Update per-step confidence EMA and decide whether to run backward.

    Ratio semantics match the reference:
        skip if  ratio > score_thresh  OR  1/ratio > score_thresh
        where ratio = mean_all / score_em
    The EMA is updated in *both* branches so it tracks the target stream.
    """

    def __init__(
        self,
        init: float,
        gamma: float,
        thresh: float,
        score_floor: float = 0.1,
        disabled: bool = False,
    ):
        self.score_em = float(init)
        self.gamma = float(gamma)
        self.thresh = float(thresh)
        self.score_floor = float(score_floor)
        self.disabled = bool(disabled)

    def step(self, scores: torch.Tensor) -> bool:
        """Return True iff the adapter should perform its gradient step.

        Empty / all-below-floor scores return False without modifying the EMA
        (nothing meaningful to update). Matches ``_teacher_pseudo``: those
        cases are the ``keep_step = False`` short-circuit.
        """
        if scores.numel() == 0:
            return False
        valid = scores > self.score_floor
        if not valid.any():
            return False

        mean_all = float(scores[valid].mean().item())
        keep_step = True
        if self.score_em > 0 and not self.disabled:
            ratio = mean_all / self.score_em
            if ratio > self.thresh or (1.0 / max(ratio, 1e-6)) > self.thresh:
                keep_step = False
        # Update EMA whether or not we ran the step.
        self.score_em = self.gamma * self.score_em + (1.0 - self.gamma) * mean_all
        return keep_step
