"""Score-EMA gate.

Returns a branch gate based on the mean teacher confidence of scores above
``score_floor`` relative to the previous score EMA. Large relative increases
or decreases close the gate. This module does not run or skip an optimizer.

The MTL adapter uses this decision for detection and CT-CL; segmentation
and CT-CR may still run. The confidence EMA here is separate from the EMA
teacher parameter update performed by the adapter.
"""
from __future__ import annotations

import torch


class ScoreEMGate:
    """Update confidence EMA and return the gate decision to the adapter.

    Ratio semantics match the reference:
        keep_step = True  if 1/thresh <= mean / score_em <= thresh
        keep_step = False otherwise (large deviation)
    With valid scores, the EMA updates whether the gate is open or closed.
    Empty / all-below-floor inputs return False without an EMA update.
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
        """Return whether confidence permits the adapter's gated branches.

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
