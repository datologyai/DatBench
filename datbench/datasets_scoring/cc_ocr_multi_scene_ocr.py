"""CC-OCR Multi-Scene OCR scoring."""

import re
from collections import Counter
from typing import Any, Dict, List


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single CC-OCR Multi-Scene OCR sample.

    Scoring methodology based on CC-OCR official evaluator:
    - Token-level F1 using Counter-based matching
    - Normalization: whitespace, lowercase, alphanumeric only
    - Matches account for token frequency

    Args:
        sample: Original sample data (from self.data)
        model_output: Model's generated text output

    Returns:
        Dictionary containing scoring results
    """
    # Get ground truth
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer", "")

    # Normalize and tokenize both
    pred_tokens = _normalize_and_tokenize(model_output)
    gt_tokens = _normalize_and_tokenize(ground_truth)

    # Compute token-level metrics
    if len(gt_tokens) == 0:
        # Empty ground truth
        recall = 1.0 if len(pred_tokens) == 0 else 0.0
        precision = 1.0 if len(pred_tokens) == 0 else 0.0
        f1 = 1.0 if len(pred_tokens) == 0 else 0.0
        right_num = 0
    elif len(pred_tokens) == 0:
        # Empty prediction
        recall = 0.0
        precision = 0.0
        f1 = 0.0
        right_num = 0
    else:
        # Counter-based matching (accounts for token frequency)
        gt_counter = Counter(gt_tokens)
        pred_counter = Counter(pred_tokens)

        # Count overlapping tokens (min of counts)
        right_num = sum(
            min(gt_counter[token], pred_counter[token]) for token in gt_counter
        )

        # Compute metrics
        recall = right_num / len(gt_tokens)
        precision = right_num / len(pred_tokens)

        # F1 with epsilon guard
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

    return {
        "score": f1,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
        "matched_tokens": right_num,
        "gt_token_count": len(gt_tokens),
        "pred_token_count": len(pred_tokens),
        "l2_category": sample.get("l2_category", ""),
        "ground_truth": ground_truth,
        "model_output": model_output,
    }


def _normalize_and_tokenize(text: str) -> List[str]:
    """Normalize and tokenize text for OCR evaluation.

    Args:
        text: Raw text string

    Returns:
        List of normalized tokens
    """
    if not isinstance(text, str):
        return []

    # Replace tabs and newlines with spaces
    text = text.replace("\t", " ").replace("\n", " ")

    # Remove special markers
    text = text.replace("###", "").replace("***", "")

    # Consolidate multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # For multi_scene_ocr: lowercase and alphanumeric only
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '', text)

    # Character-level tokenization (since we removed spaces)
    # This matches the official evaluator for alphanumeric-only mode
    tokens = list(text) if text else []

    return [t for t in tokens if t]


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CC-OCR Multi-Scene OCR metrics.

    Computes both macro-averaged (per-sample F1, then averaged) and
    micro-averaged (total tokens, then F1) metrics.

    Args:
        results: List of scored predictions from aggregate_results.py
                 Each has "score" at top level and details in "score_details"

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "accuracy": 0.0,
        }

    # Extract metrics from score_details (where aggregate_results puts them)
    f1_scores = []
    recall_scores = []
    precision_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        f1_scores.append(score_details.get("f1_score", 0.0))
        recall_scores.append(score_details.get("recall", 0.0))
        precision_scores.append(score_details.get("precision", 0.0))

    # Macro-averaged: Average per-sample metrics
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    # Micro-averaged: Aggregate all tokens first, then compute F1
    total_matched = sum(result.get("score_details", result).get("matched_tokens", 0) for result in results)
    total_gt = sum(result.get("score_details", result).get("gt_token_count", 0) for result in results)
    total_pred = sum(result.get("score_details", result).get("pred_token_count", 0) for result in results)

    if total_gt == 0:
        micro_recall = 1.0 if total_pred == 0 else 0.0
        micro_precision = 1.0 if total_pred == 0 else 0.0
    elif total_pred == 0:
        micro_recall = 0.0
        micro_precision = 0.0
    else:
        micro_recall = total_matched / total_gt
        micro_precision = total_matched / total_pred

    if micro_recall + micro_precision > 0:
        micro_f1 = 2 * micro_recall * micro_precision / (micro_recall + micro_precision)
    else:
        micro_f1 = 0.0

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "micro_precision": micro_precision,
        "accuracy": micro_f1,  # Use micro F1 as primary metric
        "score": micro_f1,  # Alias for compatibility
    }
