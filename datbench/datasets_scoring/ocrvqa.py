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

    # Compute ANLS
    anls_score_val = 0.0
    if answer:
        try:
            from anls import anls_score
            # Use official ANLS package
            anls_score_val = anls_score(
                prediction=pred_answer,
                gold_labels=[answer],  # Single answer as list
                threshold=0.5,
            )
        except ImportError:
            # Fallback: whitespace normalize
            def norm_ws(s: str) -> str:
                s = (s or "").strip().lower()
                return " ".join(s.split())

            pred_norm = norm_ws(pred_answer)
            gt_norm = norm_ws(answer)
            anls_score_val = _compute_anls(pred_norm, gt_norm)

    # Compute exact match
    def norm_ws(s: str) -> str:
        s = (s or "").strip().lower()
        return " ".join(s.split())

    pred_norm = norm_ws(pred_answer)
    gt_norm = norm_ws(answer)
    exact_match = 1.0 if pred_norm == gt_norm else 0.0

    return {
        "score": anls_score_val,  # Primary metric: ANLS
        "anls": anls_score_val,
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
    anls_scores = []
    exact_match_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        anls_scores.append(score_details.get("anls", 0.0))
        exact_match_scores.append(score_details.get("exact_match", 0.0))

    anls_avg = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
    exact_match_avg = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0

    return {
        "anls": anls_avg,
        "exact_match": exact_match_avg,
        "accuracy": anls_avg,  # ANLS is primary metric
        "score": anls_avg,  # Alias for compatibility
    }


# Helper functions

def _normalize_answer(answer: str) -> str:
    """Normalize answer for exact matching."""
    if not answer:
        return ""
    return " ".join(answer.strip().lower().split())


def _compute_anls(prediction: str, ground_truth: str) -> float:
    """Compute ANLS (threshold 0.5)."""
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0

    lev_dist = _levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    normalized_dist = lev_dist / max_len

    if normalized_dist > 0.5:
        return 0.0
    return 1.0 - normalized_dist


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
