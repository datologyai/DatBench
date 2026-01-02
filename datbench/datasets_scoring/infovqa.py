"""InfoVQA scoring."""

from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single InfoVQA sample.

    Uses ANLS (Average Normalized Levenshtein Similarity) scoring.
    Falls back to custom implementation if anls package is not available.

    Args:
        sample: Original sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Handle both original sample dict and metadata dict from evaluate script
    answer = sample.get("answer") or sample.get("ground_truth_answer")
    all_ground_truth_answers = sample.get("answers") or sample.get(
        "all_ground_truth_answers", [answer] if answer else []
    )

    # Ensure we have a list of answers
    if not isinstance(all_ground_truth_answers, list):
        all_ground_truth_answers = [all_ground_truth_answers]

    # Remove empty answers
    all_ground_truth_answers = [ans for ans in all_ground_truth_answers if ans]

    # Extract concise answer before scoring
    pred_answer = extract_final_answer(model_output)

    anls_score_val = 0.0
    if all_ground_truth_answers:
        try:
            from anls import anls_score

            # Use official ANLS package
            anls_score_val = anls_score(
                prediction=pred_answer,
                gold_labels=all_ground_truth_answers,
                threshold=0.5,
            )
        except ImportError:
            # Fall back to spec-aligned implementation: lowercase + whitespace normalize only
            def norm_ws(s: str) -> str:
                s = (s or "").strip().lower()
                return " ".join(s.split())

            det = norm_ws(pred_answer)
            for gt_answer in all_ground_truth_answers:
                gt = norm_ws(gt_answer)
                score = _compute_anls(det, gt)
                anls_score_val = max(anls_score_val, score)

    return {
        "score": anls_score_val,
        "anls": anls_score_val,  # Include anls for extract_binary_correctness
        "ground_truth": answer,
        "all_ground_truth_answers": all_ground_truth_answers,
        "model_output": model_output,
        "pred_answer": pred_answer,
    }


def _normalize_answer(answer: str) -> str:
    """Deprecated: kept for compatibility. Use whitespace normalization in fallback ANLS instead."""
    if not answer:
        return ""
    return " ".join(answer.strip().lower().split())


def _compute_anls(prediction: str, ground_truth: str) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).

    ANLS = 1 - (Levenshtein Distance / max(len(prediction), len(ground_truth)))

    If the normalized Levenshtein distance is > 0.5, ANLS = 0.
    """
    if not prediction and not ground_truth:
        return 1.0

    if not prediction or not ground_truth:
        return 0.0

    # Compute Levenshtein distance
    lev_dist = _levenshtein_distance(prediction, ground_truth)

    # Normalize by max length
    max_len = max(len(prediction), len(ground_truth))
    normalized_dist = lev_dist / max_len

    # If normalized distance > 0.5, return 0
    if normalized_dist > 0.5:
        return 0.0

    # Otherwise return 1 - normalized_dist
    return 1.0 - normalized_dist


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
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


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute InfoVQA-specific metrics.

    Args:
        results: List of result dictionaries with scores and responses

    Returns:
        Dictionary of metrics including ANLS and exact match
    """
    anls_scores = []
    exact_matches = []

    for result in results:
        score_details = result.get("score_details", result)

        # ANLS score is already computed in score_sample
        anls_scores.append(score_details.get("score", 0.0))

        # Also compute exact match
        response = score_details.get("pred_answer", "")
        all_answers = score_details.get(
            "all_ground_truth_answers", [score_details.get("ground_truth", "")]
        )

        # Check exact match
        gen_norm = _normalize_answer(response)
        exact_match = False
        for gt_answer in all_answers:
            if _normalize_answer(gt_answer) == gen_norm:
                exact_match = True
                break
        exact_matches.append(1.0 if exact_match else 0.0)

    anls_avg = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
    exact_match_avg = (
        sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    )

    return {
        "accuracy__InfoVQA-ANLS": anls_avg,
        "accuracy__InfoVQA-ExactMatch": exact_match_avg,
    }
