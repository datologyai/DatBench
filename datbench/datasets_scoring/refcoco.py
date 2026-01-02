"""RefCOCO grounding scoring."""

import re
from typing import Any, Dict, List

_RECALL_THRESHOLDS = [0.3, 0.5, 0.7]


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score RefCOCO sample using IoU."""
    pred_bbox = _parse_bbox_from_output(model_output)

    # Normalize: handle 0-1 or 0-1000 scales
    max_coord = max(abs(v) for v in pred_bbox) if pred_bbox else 0.0
    if max_coord <= 1.5:
        norm_bbox = pred_bbox
    else:
        norm_bbox = [v / 1000.0 for v in pred_bbox]
    pred_bbox = [max(0.0, min(1.0, v)) for v in norm_bbox]

    gt = sample["bbox"]  # Ground truth bbox from BeyondBench
    iou = _compute_iou(gt, pred_bbox)

    scores = {
        "iou": iou,
        "center_acc": _center_acc(gt, pred_bbox),
        "pred_bbox": pred_bbox,
        "gt_bbox": gt,
        "model_output": model_output,
    }

    for t in _RECALL_THRESHOLDS:
        scores[f"recall@{t}"] = 1.0 if iou >= t else 0.0

    scores["score"] = scores["recall@0.5"]  # Primary metric
    return scores


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute RefCOCO aggregated metrics."""
    if not results:
        return {}

    ious, center, recalls = [], [], {t: [] for t in _RECALL_THRESHOLDS}

    for r in results:
        sd = r.get("score_details", r)
        ious.append(sd.get("iou", 0.0))
        center.append(sd.get("center_acc", 0.0))
        for t in _RECALL_THRESHOLDS:
            recalls[t].append(sd.get(f"recall@{t}", 0.0))

    metrics = {
        "recall@0.5": sum(recalls[0.5]) / len(recalls[0.5]),
        "iou_mean": sum(ious) / len(ious),
        "iou": sum(ious) / len(ious),
        "center_acc": sum(center) / len(center),
    }

    for t in _RECALL_THRESHOLDS:
        metrics[f"recall@{t}"] = sum(recalls[t]) / len(recalls[t])

    return metrics


# Helper functions

def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU for [x1,y1,x2,y2] in 0-1 normalized coords."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter

    return inter / denom if denom > 0 else 0.0


def _center_acc(gt: List[float], pred: List[float]) -> float:
    """Check if predicted center falls within GT box."""
    cx = 0.5 * (pred[0] + pred[2])
    cy = 0.5 * (pred[1] + pred[3])
    return 1.0 if (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]) else 0.0


def _parse_bbox_from_output(text: str) -> List[float]:
    """Parse four floats from boxed block or bracketed list."""
    if not text:
        return [0.0, 0.0, 0.0, 0.0]

    # Prefer last \boxed{...}
    boxed_matches = re.findall(r"\\+boxed\{([^}]*)\}", text)
    candidates = []
    if boxed_matches:
        candidates.append(boxed_matches[-1])

    # Fallback: any [...] list
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    if bracket_match:
        candidates.append(bracket_match.group(0))

    for cand in candidates:
        nums = re.findall(r"-?\d+(?:\.\d+)?", cand)
        if len(nums) >= 4:
            try:
                return [float(n) for n in nums[:4]]
            except ValueError:
                continue

    return [0.0, 0.0, 0.0, 0.0]
