"""TallyQA scoring."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single TallyQA sample.

    Uses exact match for counting accuracy after extracting numeric value.
    Similar to CountBench scoring.

    Args:
        sample: Original sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Handle both original sample dict and metadata dict from evaluate script
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer")

    # Extract answer from \boxed{} format first
    extracted_output = extract_final_answer(model_output)

    # Extract numeric value from extracted output (CountBench-style normalization)
    parsed_answer = _extract_number(extracted_output)

    # Normalize ground truth to int for comparison
    gt_int = None
    if ground_truth is not None:
        try:
            if isinstance(ground_truth, str) and ground_truth.strip().isdigit():
                gt_int = int(ground_truth.strip())
            elif isinstance(ground_truth, (int, float)):
                gt_int = int(ground_truth)
        except Exception:
            gt_int = None

    # Score: exact match if parsed answer equals ground truth
    score = (
        1.0
        if parsed_answer is not None
        and gt_int is not None
        and parsed_answer == gt_int
        else 0.0
    )

    return {
        "score": score,
        "ground_truth": ground_truth,
        "parsed_answer": parsed_answer,
        "extracted_output": extracted_output,
        "model_output": model_output,
        "is_simple": sample.get("is_simple"),
    }


def _extract_number(text: str) -> Optional[int]:
    """Extract numeric value from generated text, handling both numeric and text-based numbers.

    Reuses CountBench's extraction logic for word-to-number conversion.
    """
    if not text:
        return None

    text = str(text).strip()

    # Word-to-number mapping
    word_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
    }

    # Strategy 1: Look for common ending patterns first (for chain-of-thought responses)
    final_answer_patterns = [
        r"(?:therefore|thus|hence|so),?\s*(?:the\s+)?(?:number\s+of\s+\w+\s+is\s+|answer\s+is\s+)(\w+)",
        r"(?:the\s+)?(?:total\s+)?(?:number\s+of\s+\w+\s+is\s+|count\s+is\s+)(\w+)",
        r"(?:i\s+count\s+|there\s+are\s+)(\w+)",
        r"(?:the\s+)?answer\s+is\s+(\w+)",
        r"(?:final\s+answer:?\s*)(\w+)",
        r"(?:result:?\s*)(\w+)",
    ]

    text_lower = text.lower()

    # Handle compound text numbers like "twenty-one" or "twenty one"
    tens_map = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    ones_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "zero": 0,
    }
    # Prefer the last occurrence to mirror numeric extraction behavior
    m_comp = None
    for m in re.finditer(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\-\s]+(one|two|three|four|five|six|seven|eight|nine|zero)\b",
        text_lower,
    ):
        m_comp = m
    if m_comp:
        tens = tens_map[m_comp.group(1)]
        ones = ones_map[m_comp.group(2)]
        return tens + ones
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            answer_candidate = matches[-1]  # Take the last match

            # Try to convert to number
            if answer_candidate.isdigit():
                return int(answer_candidate)
            elif answer_candidate in word_to_num:
                return word_to_num[answer_candidate]

    # Strategy 2: Look for sentence patterns
    sentence_patterns = [
        r"there\s+are\s+(\w+)\s+\w+\s+in\s+the\s+image",
        r"there\s+are\s+(\w+)\s+\w+",
        r"i\s+(?:can\s+)?see\s+(\w+)\s+\w+",
        r"contains\s+(\w+)\s+\w+",
        r"shows\s+(\w+)\s+\w+",
    ]

    for pattern in sentence_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            answer_candidate = matches[-1]

            if answer_candidate.isdigit():
                return int(answer_candidate)
            elif answer_candidate in word_to_num:
                return word_to_num[answer_candidate]

    # Strategy 3: Look for any numeric digits (fallback)
    # Guard against absurdly long numeric tokens (LLM failures producing hundreds of digits)
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        num_token = numbers[-1]
        if len(num_token) <= 9:  # keep within a reasonable range for counts
            try:
                return int(num_token)
            except ValueError:
                pass

    # Strategy 4: Look for text-based numbers anywhere
    text_numbers = []
    for word in text_lower.split():
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in word_to_num:
            text_numbers.append(word_to_num[clean_word])

    if text_numbers:
        return text_numbers[-1]

    return None


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute TallyQA-specific metrics.

    Computes overall accuracy, simple subset accuracy, complex subset accuracy,
    within-1 accuracy, within-3 accuracy, parse rate, and MAE.

    Args:
        results: List of result dictionaries with scores

    Returns:
        Dictionary of metrics
    """
    n_total = len(results)
    n_correct_from_parse = 0
    n_parsed = 0
    absolute_errors: List[float] = []

    # Prefer using precomputed scores if present
    score_values: List[float] = []

    # Separate lists for simple and complex subsets
    simple_scores: List[float] = []
    complex_scores: List[float] = []
    simple_absolute_errors: List[float] = []
    complex_absolute_errors: List[float] = []

    for result in results:
        # Get score_details (which contains the scoring info)
        score_details = result.get("score_details", result)

        # 1) Gather ground truth
        ground_truth = score_details.get("ground_truth")
        if ground_truth is None:
            ground_truth = result.get("ground_truth_answer")

        # Get is_simple flag
        is_simple = score_details.get("is_simple")
        if is_simple is None:
            is_simple = result.get("is_simple")

        # Normalize ground truth to int when possible
        gt_int = None
        try:
            if isinstance(ground_truth, str) and ground_truth.strip().isdigit():
                gt_int = int(ground_truth.strip())
            elif isinstance(ground_truth, (int, float)):
                gt_int = int(ground_truth)
        except Exception:
            gt_int = None

        # 2) Prefer precomputed score if present
        if "score" in score_details:
            try:
                score_val = float(score_details.get("score", 0.0))
                score_values.append(score_val)
                # Add to subset lists
                if is_simple is True:
                    simple_scores.append(score_val)
                elif is_simple is False:
                    complex_scores.append(score_val)
            except Exception:
                score_values.append(0.0)
                if is_simple is True:
                    simple_scores.append(0.0)
                elif is_simple is False:
                    complex_scores.append(0.0)

        # 3) Determine parsed answer for parse rate / MAE
        parsed_answer = score_details.get("parsed_answer")
        if parsed_answer is None:
            extracted_output = score_details.get("extracted_output")
            if isinstance(extracted_output, (int, float)):
                parsed_answer = int(extracted_output)
            elif isinstance(extracted_output, str):
                parsed_answer = _extract_number(extracted_output)
        if parsed_answer is None:
            text_out = score_details.get("model_output", "")
            parsed_answer = _extract_number(text_out)

        # 4) Update parse-based stats
        if parsed_answer is not None:
            # Sanitize unreasonable parsed values
            try:
                parsed_int = int(parsed_answer)
            except Exception:
                parsed_int = None

            if parsed_int is not None and (parsed_int < 0 or parsed_int > 1_000_000):
                parsed_int = None

            if parsed_int is not None:
                n_parsed += 1
                if gt_int is not None:
                    if parsed_int == int(gt_int):
                        n_correct_from_parse += 1
                    abs_error = abs(parsed_int - int(gt_int))
                    if abs_error <= 1_000_000:
                        absolute_errors.append(abs_error)
                        # Add to subset error lists
                        if is_simple is True:
                            simple_absolute_errors.append(abs_error)
                        elif is_simple is False:
                            complex_absolute_errors.append(abs_error)

    # Overall accuracy: prefer precomputed scores if available for all samples
    if len(score_values) == n_total and n_total > 0:
        accuracy = float(sum(score_values) / n_total)
    else:
        accuracy = float(n_correct_from_parse) / n_total if n_total > 0 else 0.0

    # Simple subset accuracy
    accuracy_simple = 0.0
    n_simple_total = len(
        [
            r
            for r in results
            if r.get("score_details", {}).get("is_simple") is True
            or r.get("is_simple") is True
        ]
    )
    if simple_scores and len(simple_scores) == n_simple_total:
        accuracy_simple = float(sum(simple_scores) / len(simple_scores))
    elif simple_absolute_errors and n_simple_total > 0:
        # Compute from parse-based stats if scores not available
        n_simple_correct = sum(1 for e in simple_absolute_errors if e == 0)
        accuracy_simple = float(n_simple_correct / n_simple_total)

    # Complex subset accuracy
    accuracy_complex = 0.0
    n_complex_total = len(
        [
            r
            for r in results
            if r.get("score_details", {}).get("is_simple") is False
            or r.get("is_simple") is False
        ]
    )
    if complex_scores and len(complex_scores) == n_complex_total:
        accuracy_complex = float(sum(complex_scores) / len(complex_scores))
    elif complex_absolute_errors and n_complex_total > 0:
        # Compute from parse-based stats if scores not available
        n_complex_correct = sum(1 for e in complex_absolute_errors if e == 0)
        accuracy_complex = float(n_complex_correct / n_complex_total)

    parse_rate = float(n_parsed) / n_total if n_total > 0 else 0.0
    mae = float(np.mean(absolute_errors)) if absolute_errors else 0.0
    within_1 = (
        sum(1 for e in absolute_errors if e <= 1) / n_total if n_total > 0 else 0.0
    )
    within_3 = (
        sum(1 for e in absolute_errors if e <= 3) / n_total if n_total > 0 else 0.0
    )

    return {
        "accuracy__TallyQA-Accuracy": accuracy,
        "accuracy__TallyQA-AccuracySimple": accuracy_simple,
        "accuracy__TallyQA-AccuracyComplex": accuracy_complex,
        "accuracy__TallyQA-ParseRate": parse_rate,
        "accuracy__TallyQA-MAE": mae,
        "accuracy__TallyQA-Within1": within_1,
        "accuracy__TallyQA-Within3": within_3,
    }
