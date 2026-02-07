"""MME-RealWorld scoring."""

import re
from typing import Any, Dict, List


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score an MME-RealWorld sample.

    Extracts letter choice and performs exact match against ground truth.

    Args:
        sample: Sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing score
    """
    ground_truth = (
        sample.get("answer")
        or sample.get("ground_truth_answer")
        or sample.get("mcq_answer_letter")
        or ""
    )
    choices = (
        sample.get("options")
        or sample.get("candidate_answers")
        or sample.get("choices")
        or []
    )

    # Extract letter from model output
    extracted_letter = _extract_letter(model_output, choices)

    # Exact match
    correct = (extracted_letter.upper() == ground_truth.upper())

    return {
        "score": 1.0 if correct else 0.0,
        "extracted_letter": extracted_letter,
        "ground_truth": ground_truth,
        "model_output": model_output,
    }


def _extract_letter(text: str, choices: List[str]) -> str:
    """Extract letter (A-E) from model output.

    Based on lmms-eval utils.py:extract_characters_regex()
    """
    if isinstance(text, dict):
        text = ""

    text = text.strip()

    # Remove common answer prefixes
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for prefix in answer_prefixes:
        text = text.replace(prefix, "")

    # Prefer explicit boxed markers at the end of CoT
    boxed_patterns = [
        r"\\+boxed\{\s*([A-Ea-e])\s*\}",
        r"<\|begin_of_box\|>\s*([A-Ea-e])\s*<\|end_of_box\|>",
    ]
    for pat in boxed_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            if val:
                return val.upper()

    # Choose the first standalone letter (answer at start of response)
    letter_tokens = re.findall(r"(?<![A-Za-z0-9])([A-Ea-e])(?![A-Za-z0-9])", text)
    if letter_tokens:
        return letter_tokens[0].upper()

    # If text is too long and still no letter, return empty
    if len(text.split()) > 10 and not re.search("[ABCDE]", text):
        return ""

    # Fallback: try to match the entire (lowercased) text to an option; otherwise empty
    for choice in choices:
        if text.lower() in choice.lower():
            if len(choice) >= 2 and choice[1] in "ABCDE":
                return choice[1]
    return ""


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute MME-RealWorld metrics.

    Args:
        results: List of scored predictions

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {"accuracy": 0.0}

    # Extract scores
    scores = []
    for result in results:
        score_details = result.get("score_details", result)
        scores.append(score_details.get("score", 0.0))

    accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "accuracy": accuracy,
        "score": accuracy,
    }
