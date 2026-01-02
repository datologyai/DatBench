"""CharXiv scoring.

Note: CharXiv officially uses LM judge-based scoring. This module provides
a fallback implementation with direct answer matching.
"""

from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single CharXiv sample.

    Args:
        sample: Sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    question_type = sample.get("question_type") or sample.get("metadata", {}).get(
        "question_type", "descriptive"
    )
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer", "")

    pred_answer = extract_final_answer(model_output)
    score = 0.0

    if question_type == "descriptive":
        score = _score_descriptive_simplified(pred_answer, ground_truth)
    else:
        reasoning_q_source = sample.get("reasoning_q_source") or sample.get(
            "metadata", {}
        ).get("reasoning_q_source", 1)
        score = _score_reasoning_simplified(
            pred_answer, ground_truth, reasoning_q_source
        )

    return {
        "score": score,
        "ground_truth": ground_truth,
        "model_output": model_output,
        "pred_answer": pred_answer,
        "question_type": question_type,
    }


def _score_descriptive_simplified(
    prediction: str, ground_truth: str
) -> float:
    """Simplified scoring for descriptive questions.

    This is a basic implementation. Full scoring should use GPT-4 with rubrics.
    """
    if not prediction or not ground_truth:
        return 0.0

    # Normalize answers
    pred_norm = prediction.strip().lower()
    gt_norm = ground_truth.strip().lower()

    # Handle "Not Applicable"
    if gt_norm == "not applicable":
        return 1.0 if "not applicable" in pred_norm else 0.0

    # Basic exact match (case-insensitive)
    if pred_norm == gt_norm:
        return 1.0

    # Try numeric comparison
    try:
        pred_num = float(pred_norm.replace(",", "").replace("%", ""))
        gt_num = float(gt_norm.replace(",", "").replace("%", ""))
        if abs(pred_num - gt_num) < 1e-6:
            return 1.0
    except ValueError:
        pass

    return 0.0


def _score_reasoning_simplified(
    prediction: str, ground_truth: str, reasoning_q_source: int
) -> float:
    """Simplified scoring for reasoning questions.

    This is a basic implementation. Full scoring should use GPT-4 with category-specific rules.
    """
    if not prediction or not ground_truth:
        return 0.0

    pred_norm = prediction.strip().lower()
    gt_norm = ground_truth.strip().lower()

    if reasoning_q_source in [3, 4]:
        # Numeric: exact match or equivalent notation
        try:
            pred_num = float(pred_norm.replace(",", "").replace("%", ""))
            gt_num = float(gt_norm.replace(",", "").replace("%", ""))
            if abs(pred_num - gt_num) < 1e-6:
                return 1.0
        except ValueError:
            pass
        return 0.0
    else:
        # Text-based: basic matching
        if pred_norm == gt_norm:
            return 1.0
        # Check if prediction contains ground truth or vice versa
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return 1.0
        return 0.0


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CharXiv-specific metrics.

    Args:
        results: List of result dictionaries with scores

    Returns:
        Dictionary of metrics
    """
    scores = []
    descriptive_scores = []
    reasoning_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        score = score_details.get("score", 0.0)
        scores.append(score)

        question_type = score_details.get("question_type", "descriptive")
        if question_type == "descriptive":
            descriptive_scores.append(score)
        else:
            reasoning_scores.append(score)

    accuracy = sum(scores) / len(scores) if scores else 0.0

    metrics = {
        "accuracy__CharXiv-Overall": accuracy,
    }

    if descriptive_scores:
        metrics["accuracy__CharXiv-Descriptive"] = sum(descriptive_scores) / len(
            descriptive_scores
        )

    if reasoning_scores:
        metrics["accuracy__CharXiv-Reasoning"] = sum(reasoning_scores) / len(
            reasoning_scores
        )

    return metrics
