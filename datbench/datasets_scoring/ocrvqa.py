"""OCR-VQA scoring.


"""

from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single OCR-VQA sample using ANLS.

    Args:
        sample: Sample data with 'answer' (single answer, not list)
        model_output: Model's generated output

    Returns:
        Dictionary with score, anls, exact_match, etc.
    """
    # Get ground truth (single answer for OCR-VQA)
    answer = sample.get("answer", "")

    # Extract concise answer
    pred_answer = extract_final_answer(model_output)

    # Compute ANLS variants:
    # - raw/continuous similarity (no threshold)
    # - thresholded ANLS@0.5 (standard ANLS clipping)
    # - binary threshold@0.5 accuracy (new explicit metric)
    pred_norm = _normalize_answer(pred_answer)
    gt_norm = _normalize_answer(answer)
    anls_raw = _compute_anls_raw(pred_norm, gt_norm) if answer else 0.0
    anls_thresholded = _apply_anls_threshold(anls_raw)
    threshold_0_5_accuracy = _threshold_at_0_5(anls_raw)

    # Compute exact match
    exact_match = 1.0 if pred_norm == gt_norm else 0.0

    return {
        # Keep score as continuous metric so aggregate accuracy captures
        # partial-correctness instead of hard-thresholding.
        "score": anls_raw,
        "anls": anls_raw,
        "anls_thresholded_0_5": anls_thresholded,
        "threshold_0_5_accuracy": threshold_0_5_accuracy,
        "exact_match": exact_match,
        "ground_truth": answer,
        "model_output": model_output,
        "pred_answer": pred_answer,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute OCR-VQA metrics.

    Args:
        results: List of score_details dicts from score_sample

    Returns:
        Dictionary with anls, exact_match, accuracy, score
    """
    anls_raw_scores = []
    anls_thresholded_scores = []
    threshold_0_5_scores = []
    exact_match_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        anls_raw_scores.append(score_details.get("anls", score_details.get("score", 0.0)))
        anls_thresholded_scores.append(score_details.get("anls_thresholded_0_5", 0.0))
        threshold_0_5_scores.append(score_details.get("threshold_0_5_accuracy", 0.0))
        exact_match_scores.append(score_details.get("exact_match", 0.0))

    anls_raw_avg = sum(anls_raw_scores) / len(anls_raw_scores) if anls_raw_scores else 0.0
    anls_thresholded_avg = (
        sum(anls_thresholded_scores) / len(anls_thresholded_scores)
        if anls_thresholded_scores
        else 0.0
    )
    threshold_0_5_avg = (
        sum(threshold_0_5_scores) / len(threshold_0_5_scores)
        if threshold_0_5_scores
        else 0.0
    )
    exact_match_avg = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0

    return {
        "anls": anls_raw_avg,
        "anls_thresholded_0_5": anls_thresholded_avg,
        "threshold_0_5_accuracy": threshold_0_5_avg,
        "exact_match": exact_match_avg,
        "accuracy": anls_raw_avg,  # Primary metric: continuous ANLS
        "score": anls_raw_avg,  # Alias for compatibility
    }


# Helper functions

def _normalize_answer(answer: str) -> str:
    """Normalize answer for exact matching."""
    if not answer:
        return ""
    return " ".join(answer.strip().lower().split())


def _compute_anls_raw(prediction: str, ground_truth: str) -> float:
    """Compute raw ANLS similarity without threshold clipping."""
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0

    lev_dist = _levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    normalized_dist = lev_dist / max_len

    return 1.0 - normalized_dist


def _apply_anls_threshold(raw_score: float, threshold: float = 0.5) -> float:
    """Apply standard ANLS thresholding."""
    return raw_score if raw_score >= threshold else 0.0


def _threshold_at_0_5(raw_score: float) -> float:
    """Binary threshold metric requested for tracking hard 0/1 accuracy."""
    return 1.0 if raw_score >= 0.5 else 0.0


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
