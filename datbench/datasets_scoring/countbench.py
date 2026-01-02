"""CountBench scoring."""

import re
from typing import Any, Dict, List, Optional
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score CountBench sample - exact numeric matching."""
    ground_truth_raw = sample.get("answer") or sample.get("ground_truth_answer")
    extracted_output = extract_final_answer(model_output)
    parsed_answer = _extract_number(extracted_output)

    # Normalize ground truth to int for comparison
    ground_truth = None
    try:
        if isinstance(ground_truth_raw, str):
            ground_truth = int(ground_truth_raw.strip())
        elif isinstance(ground_truth_raw, (int, float)):
            ground_truth = int(ground_truth_raw)
    except (ValueError, AttributeError):
        ground_truth = ground_truth_raw

    score = 1.0 if parsed_answer is not None and parsed_answer == ground_truth else 0.0

    return {
        "score": score,
        "ground_truth": ground_truth,
        "parsed_answer": parsed_answer,
        "model_output": model_output,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CountBench metrics - ."""
    import numpy as np

    n_total = len(results)
    n_correct_from_parse = 0
    n_parsed = 0
    absolute_errors = []
    score_values = []

    for result in results:
        # Get ground truth
        sd = result.get("score_details", result)
        ground_truth = sd.get("ground_truth")

        # Normalize GT to int
        gt_int = None
        try:
            if isinstance(ground_truth, str) and ground_truth.strip().isdigit():
                gt_int = int(ground_truth.strip())
            elif isinstance(ground_truth, (int, float)):
                gt_int = int(ground_truth)
        except Exception:
            gt_int = None

        # Get precomputed score
        if "score" in sd:
            try:
                score_values.append(float(sd.get("score", 0.0)))
            except Exception:
                score_values.append(0.0)

        # Get parsed answer
        parsed_answer = sd.get("parsed_answer")

        # Update parse stats
        if parsed_answer is not None:
            try:
                parsed_int = int(parsed_answer)
            except Exception:
                parsed_int = None

            # Sanitize unreasonable values
            if parsed_int is not None and (parsed_int < 0 or parsed_int > 1_000_000):
                parsed_int = None

            if parsed_int is not None:
                n_parsed += 1
                if gt_int is not None:
                    if parsed_int == int(gt_int):
                        n_correct_from_parse += 1
                    diff = abs(parsed_int - int(gt_int))
                    if diff <= 1_000_000:
                        absolute_errors.append(diff)

    # Compute metrics
    accuracy = float(sum(score_values) / n_total) if len(score_values) == n_total and n_total > 0 else (float(n_correct_from_parse) / n_total if n_total > 0 else 0.0)
    parse_rate = float(n_parsed) / n_total if n_total > 0 else 0.0
    mae = float(np.mean(absolute_errors)) if absolute_errors else 0.0
    within_1 = sum(1 for e in absolute_errors if e <= 1) / n_total if n_total > 0 else 0.0
    within_3 = sum(1 for e in absolute_errors if e <= 3) / n_total if n_total > 0 else 0.0

    return {
        "accuracy__CountBench-Accuracy": accuracy,
        "accuracy__CountBench-ParseRate": parse_rate,
        "accuracy__CountBench-MAE": mae,
        "accuracy__CountBench-Within1": within_1,
        "accuracy__CountBench-Within3": within_3,
    }


def _extract_number(text: str) -> Optional[int]:
    """Extract number from text with word-to-number support."""
    if not text:
        return None

    # Try digit first
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        try:
            return int(numbers[-1])
        except:
            pass

    # Word to number
    word_map = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for word, num in word_map.items():
        if word in text.lower():
            return num

    return None
