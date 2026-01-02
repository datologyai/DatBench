"""MathVision scoring."""

from typing import Any, Dict, List
from .evaluation_utils.mathvision_utils import score_prediction, aggregate_correct_list


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a MathVision sample

    Args:
        sample: Sample data with 'answer' and 'options'
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    option_pairs = sample.get("options_struct") or _build_option_pairs(
        sample.get("options") or []
    )
    answer = sample.get("correct_choice_letter") or sample.get("answer") or ""
    doc = {
        "answer": str(answer).strip(),
        "options": [p["text"] for p in option_pairs],
    }

    scored = score_prediction(doc, model_output)
    return {
        "score": 1.0 if scored.get("correct") else 0.0,
        "pred_answer": model_output,
        "normalized_prediction": scored.get("normalized_prediction", ""),
        "normalized_target": scored.get("normalized_target", ""),
    }


def _build_option_pairs(options: List[Any]) -> List[Dict[str, str]]:
    """Build option pairs from list."""
    pairs: List[Dict[str, str]] = []
    for idx, raw in enumerate(options):
        label = chr(ord("A") + idx)
        value = str(raw).strip()
        pairs.append({"label": label, "text": value if value else label})
    return pairs


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute MathVision metrics.

    Args:
        results: List of result dictionaries with scores

    Returns:
        Dictionary of metrics
    """
    correct = []
    for result in results:
        score_details = result.get("score_details", result)
        correct.append(score_details.get("score", 0.0) == 1.0)

    acc_frac = aggregate_correct_list(correct)
    return {"accuracy": acc_frac}
