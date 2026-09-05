"""EMA (Mean-Teacher) update.

Matches the reference detectron2 CTCMT_MTL implementation:
    teacher_param <- decay * teacher_param + (1 - decay) * student_param
    non-float buffers are copied verbatim from the student.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class EMAUpdater:
    """Applies EMA once per adaptation step."""

    def __init__(self, decay: float):
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, teacher: nn.Module, student: nn.Module) -> None:
        d = self.decay
        s_state = student.state_dict()
        for k, v in teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s_state[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s_state[k])
