"""CountBench scoring."""

import re
from typing import Any, Dict, List, Optional
from .evaluation_utils.erma_utils import extract_final_answer


# Number words the extractor recognises (single tokens only; compound forms
# such as "twenty-one" are not resolved).
_WORD_TO_NUM = {
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

# Strategy 1: explicit final-answer phrasing typical of chain-of-thought responses.
_FINAL_ANSWER_PATTERNS = [
    r"(?:therefore|thus|hence|so),?\s*(?:the\s+)?(?:number\s+of\s+\w+\s+is\s+|answer\s+is\s+)(\w+)",
    r"(?:the\s+)?(?:total\s+)?(?:number\s+of\s+\w+\s+is\s+|count\s+is\s+)(\w+)",
    r"(?:i\s+count\s+|there\s+are\s+)(\w+)",
    r"(?:the\s+)?answer\s+is\s+(\w+)",
    r"(?:final\s+answer:?\s*)(\w+)",
    r"(?:result:?\s*)(\w+)",
]

# Strategy 2: descriptive sentences that state a count.
_SENTENCE_PATTERNS = [
    r"there\s+are\s+(\w+)\s+\w+\s+in\s+the\s+image",
    r"there\s+are\s+(\w+)\s+\w+",
    r"i\s+(?:can\s+)?see\s+(\w+)\s+\w+",
    r"contains\s+(\w+)\s+\w+",
    r"shows\s+(\w+)\s+\w+",
]

# Strategy 3: a bare digit token longer than this is degenerate output (e.g. a
# runaway digit string), not a count, and is ignored.
_MAX_NUMERIC_TOKEN_LENGTH = 9


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
        "extracted_output": extracted_output,
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


def _candidate_to_int(candidate: str) -> Optional[int]:
    """Convert a regex-captured token to an int if it is digits or a number word."""
    if candidate.isdigit():
        return int(candidate)
    return _WORD_TO_NUM.get(candidate)


def _extract_number(text: str) -> Optional[int]:
    """Extract the answered count from text, handling digits and number words.

    Strategies are tried in order and the first that yields a value wins:
    explicit final-answer phrasing, descriptive count sentences, the last bare
    digit token, then the last number word anywhere in the text. Within a
    strategy the last match is used so that a concluding statement outranks
    intermediate reasoning.
    """
    if not text:
        return None

    text = str(text).strip()
    text_lower = text.lower()

    # Strategy 1: final-answer phrasing (chain-of-thought conclusions)
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            value = _candidate_to_int(matches[-1])
            if value is not None:
                return value

    # Strategy 2: descriptive count sentences
    for pattern in _SENTENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            value = _candidate_to_int(matches[-1])
            if value is not None:
                return value

    # Strategy 3: last bare digit token, ignoring runaway digit strings
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        num_token = numbers[-1]
        if len(num_token) <= _MAX_NUMERIC_TOKEN_LENGTH:
            try:
                return int(num_token)
            except ValueError:
                pass

    # Strategy 4: last number word, matched as a whole token after stripping punctuation
    text_numbers = []
    for word in text_lower.split():
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in _WORD_TO_NUM:
            text_numbers.append(_WORD_TO_NUM[clean_word])

    if text_numbers:
        return text_numbers[-1]

    return None
