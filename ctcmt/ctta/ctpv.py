"""Cross-Task Pseudo-label Verification (CTPV).

Reject a teacher-detected box of class c when the teacher's semantic-seg
head assigns fewer than ``thresh`` of the pixels inside the box to the
detection class's seg-taxonomy counterpart (``det_to_seg_cls[c]``).

Matches ``CTCMT_MTL._ctpv_filter`` in the reference — hard-argmax agreement.
Soft-probability CTPV is deliberately not introduced here; keep the
reproduction pass identical to the published method.
"""
from __future__ import annotations

from typing import Sequence

import torch


class CTPVFilter:
    def __init__(self, det_to_seg_cls: Sequence[int], thresh: float):
        self.det_to_seg_cls = tuple(int(x) for x in det_to_seg_cls)
        self.thresh = float(thresh)

    @torch.no_grad()
    def filter(self, instances, teacher_seg_probs: torch.Tensor):
        """teacher_seg_probs shape: ``(1, K, Hs, Ws)`` in probability space."""
        from detectron2.structures import Boxes, Instances

        if len(instances) == 0 or teacher_seg_probs is None:
            return instances

        _, K, Hs, Ws = teacher_seg_probs.shape
        img_h, img_w = instances.image_size
        sx = Ws / max(img_w, 1)
        sy = Hs / max(img_h, 1)

        keep = []
        boxes = instances.pred_boxes.tensor
        classes = instances.pred_classes
        for j in range(len(instances)):
            x1, y1, x2, y2 = boxes[j].tolist()
            c = int(classes[j].item())
            if not (0 <= c < len(self.det_to_seg_cls)):
                keep.append(True)
                continue
            seg_c = self.det_to_seg_cls[c]
            if seg_c >= K:
                keep.append(True)
                continue
            x1i = max(int(round(x1 * sx)), 0)
            y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), Ws)
            y2i = min(int(round(y2 * sy)), Hs)
            if x2i <= x1i or y2i <= y1i:
                keep.append(False)
                continue
            region = teacher_seg_probs[0, :, y1i:y2i, x1i:x2i]
            agreement = float((region.argmax(dim=0) == seg_c).float().mean().item())
            keep.append(agreement >= self.thresh)

        if all(keep):
            return instances

        keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.bool)
        new_inst = Instances(instances.image_size)
        new_inst.pred_boxes = Boxes(boxes[keep_t])
        new_inst.pred_classes = classes[keep_t]
        new_inst.scores = instances.scores[keep_t]
        return new_inst
