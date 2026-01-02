"""Pixmo Points Eval scoring.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single Pixmo Points Eval sample.

    Uses the Jonker-Volgenant algorithm for point assignment and segmentation masks for scoring.

    Args:
        sample: Sample data with 'points', 'masks', 'has_target'
        model_output: Model's generated output

    Returns:
        Dictionary with score, precision, recall, f1, etc.
    """
    # Parse predicted coordinates from the generated answer
    predicted_points = _parse_coordinates_from_response(model_output)

    raw_ground_truth_points = sample.get("points", [])
    ground_truth_points = _normalize_points(raw_ground_truth_points, gt=True)
    masks = sample.get("masks", [])
    has_target = sample.get("has_target", len(ground_truth_points) > 0)

    # Initialize scoring variables
    score = 0.0
    precision = 0.0
    recall = 0.0

    # Case 1: No target object in the image
    if not has_target:
        # Check if model correctly responds that there's no target
        no_target_phrases = [
            "no target", "no such object", "no object", "no object found",
            "no target found", "no target present", "target not found",
            "object not found", "not present", "is not present",
            "absent", "missing", "cannot find", "cannot locate",
            "can't find", "can't locate",
        ]
        response_lower = model_output.lower()
        if any(phrase in response_lower for phrase in no_target_phrases):
            score = 1.0  # Correct no-target response

    # Case 2: Target object is present
    elif ground_truth_points and predicted_points:
        # Use Jonker-Volgenant algorithm to assign predicted points to ground truth points
        try:
            assignments = _assign_points_jonker_volgenant(
                predicted_points, ground_truth_points
            )

            # Calculate precision: fraction of predicted points within segmentation masks
            true_positives = 0
            for pred_idx, gt_idx in assignments:
                if pred_idx < len(predicted_points) and gt_idx < len(masks):
                    pred_point = predicted_points[pred_idx]
                    mask = masks[gt_idx]
                    if _point_in_mask(pred_point, mask):
                        true_positives += 1

            precision = (
                true_positives / len(predicted_points)
                if len(predicted_points) > 0
                else 0.0
            )

            # Calculate recall: fraction of ground truth points covered by predictions
            covered_gt_points = 0
            for pred_idx, gt_idx in assignments:
                if pred_idx < len(predicted_points) and gt_idx < len(masks):
                    pred_point = predicted_points[pred_idx]
                    mask = masks[gt_idx]
                    if _point_in_mask(pred_point, mask):
                        covered_gt_points += 1

            recall = (
                covered_gt_points / len(ground_truth_points)
                if len(ground_truth_points) > 0
                else 0.0
            )

            # Compute F1 score
            score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        except Exception:
            # Assignment failed, score is 0
            score = 0.0

    return {
        "score": score,
        "predicted_points": predicted_points,
        "ground_truth_points": ground_truth_points,
        "has_target": has_target,
        "precision": precision,
        "recall": recall,
        "model_output": model_output,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregated metrics from score_sample results."""
    if not results:
        return {}

    precisions, recalls, f1s = [], [], []
    no_target_correct = []
    parsing_failures = []

    for r in results:
        sd = r.get("score_details", r)
        precisions.append(sd.get("precision", 0.0))
        recalls.append(sd.get("recall", 0.0))

        # F1 can be computed from precision/recall
        p, r = sd.get("precision", 0.0), sd.get("recall", 0.0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)

        # No target accuracy
        if not sd.get("has_target", True):
            no_target_correct.append(1.0 if sd.get("score", 0.0) >= 0.5 else 0.0)

        # Parsing failure: has target but no predicted points
        if sd.get("has_target") and not sd.get("predicted_points"):
            parsing_failures.append(1.0)
        else:
            parsing_failures.append(0.0)

    metrics = {
        "precision__PixmoPointsEval-Precision": sum(precisions) / len(precisions),
        "recall__PixmoPointsEval-Recall": sum(recalls) / len(recalls),
        "f1__PixmoPointsEval-F1": sum(f1s) / len(f1s),
        "accuracy__PixmoPointsEval-NoTargetAccuracy": (
            sum(no_target_correct) / len(no_target_correct) if no_target_correct else 0.0
        ),
        "accuracy__PixmoPointsEval-ParsingFailureRate": (
            sum(parsing_failures) / len(parsing_failures)
        ),
    }
    return metrics


# Helper functions (all private, extracted from PixmoPointsEvalDataset)

def _parse_coordinates_from_response(response: str) -> List[Tuple[float, float]]:
    """Parse coordinate pairs from model response."""
    coordinates = []
    response = response.strip()

    # Pattern 0: try boxed coordinates first (e.g., \boxed{120, 870})
    boxed_pattern = r"\\+boxed\{([^}]*)\}"
    boxed_matches = re.findall(boxed_pattern, response)
    for content in reversed(boxed_matches):
        coords = _parse_coords_from_string(content)
        if coords:
            return coords

    # Pattern 1: (x, y) - parentheses
    parentheses_pattern = r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)"
    matches = re.findall(parentheses_pattern, response)
    if matches:
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                coords = _normalize_pred_point((x, y))
                if coords is not None:
                    coordinates.append(coords)
            except (ValueError, IndexError):
                continue
        if coordinates:
            return coordinates

    # Pattern 2: [x, y] - square brackets
    brackets_pattern = r"\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]"
    matches = re.findall(brackets_pattern, response)
    if matches:
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                coords = _normalize_pred_point((x, y))
                if coords is not None:
                    coordinates.append(coords)
            except (ValueError, IndexError):
                continue
        if coordinates:
            return coordinates

    # Pattern 3: x, y - comma separated
    comma_pattern = r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
    matches = re.findall(comma_pattern, response)
    if matches:
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                coords = _normalize_pred_point((x, y))
                if coords is not None:
                    coordinates.append(coords)
            except (ValueError, IndexError):
                continue

    return coordinates


def _parse_coords_from_string(text: str) -> List[Tuple[float, float]]:
    """Parse coordinate pairs from a raw string (used inside boxed blocks)."""
    coords = []
    patterns = [
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+([0-9]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        for x_str, y_str in re.findall(pat, text):
            x, y = float(x_str), float(y_str)
            normalized = _normalize_pred_point((x, y))
            if normalized is not None:
                coords.append(normalized)
    return coords


def _normalize_pred_point(point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Normalize model output point to [0,1] supporting two scales: [0,1] or 0–1000."""
    x, y = point
    maxv = max(abs(x), abs(y))
    if maxv <= 1.0:
        norm = (x, y)
    else:
        if min(x, y) < 0.0:
            return None
        norm = (min(x, 1000.0) / 1000.0, min(y, 1000.0) / 1000.0)
    return (
        max(0.0, min(1.0, norm[0])),
        max(0.0, min(1.0, norm[1])),
    )


def _normalize_gt_point(point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Normalize ground-truth point to [0,1]; GT may be in [0,1] or 0–100 (percent)."""
    x, y = point
    maxv = max(abs(x), abs(y))
    if maxv <= 1.0:
        norm = (x, y)
    elif maxv <= 100.0:
        norm = (x / 100.0, y / 100.0)
    else:
        if min(x, y) < 0.0:
            return None
        norm = (min(x, 1000.0) / 1000.0, min(y, 1000.0) / 1000.0)
    return (
        max(0.0, min(1.0, norm[0])),
        max(0.0, min(1.0, norm[1])),
    )


def _normalize_points(points: List[Any], gt: bool = False) -> List[Tuple[float, float]]:
    """Normalize a list of points to [0,1]."""
    normalized = []
    for pt in points:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            xy = (float(pt["x"]), float(pt["y"]))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                xy = (float(pt[0]), float(pt[1]))
            except (ValueError, TypeError):
                continue
        else:
            continue

        norm = _normalize_gt_point(xy) if gt else _normalize_pred_point(xy)
        if norm is not None:
            normalized.append(norm)
    return normalized


def _assign_points_jonker_volgenant(
    predicted: List[Tuple[float, float]],
    ground_truth: List[Tuple[float, float]]
) -> List[Tuple[int, int]]:
    """Assign predicted points to GT using Jonker-Volgenant (Hungarian) algorithm."""
    if not predicted or not ground_truth:
        return []

    # Build cost matrix: Euclidean distance between all pairs
    cost_matrix = np.zeros((len(predicted), len(ground_truth)))
    for i, pred_pt in enumerate(predicted):
        for j, gt_pt in enumerate(ground_truth):
            dx = pred_pt[0] - gt_pt[0]
            dy = pred_pt[1] - gt_pt[1]
            cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def _point_in_mask(point: Tuple[float, float], mask: List[List[bool]]) -> bool:
    """Check if a normalized point is within a segmentation mask."""
    if not mask or not mask[0]:
        return False

    mask_height = len(mask)
    mask_width = len(mask[0])

    # Convert normalized coordinates to mask coordinates
    x_mask = int(point[0] * mask_width)
    y_mask = int(point[1] * mask_height)

    # Check bounds
    if y_mask < 0 or y_mask >= mask_height or x_mask < 0 or x_mask >= mask_width:
        return False

    return mask[y_mask][x_mask]
