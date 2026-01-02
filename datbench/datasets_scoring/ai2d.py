"""AI2D scoring."""

import re
import numpy as np
from typing import Any, Dict, List, Optional
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score an AI2D sample.

    Args:
        sample: Original sample data with 'answer' (correct choice index) and 'answer_choices'
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Get ground truth and choices
    answer_label = sample.get("answer") or sample.get("answer_label") or sample.get("ground_truth_answer")
    answer_choices = sample.get("answer_choices", [])

    # Parse the model's response to get predicted choice
    parsed_response = _parse_multi_choice_response(model_output, answer_choices)

    # Convert choice letter to index (A=0, B=1, C=2, D=3)
    if parsed_response and parsed_response in "ABCD":
        predicted_index = ord(parsed_response) - ord("A")
    else:
        predicted_index = None

    # Check if correct
    is_correct = (
        predicted_index is not None and predicted_index == answer_label
    )

    return {
        "score": 1.0 if is_correct else 0.0,
        "extracted_output": parsed_response,
        "predicted_index": predicted_index,
        "is_correct": is_correct,
        "pred_answer": parsed_response,
        "ground_truth": answer_label,
    }


def _parse_multi_choice_response(
    response: str, answer_choices: List[str]
) -> Optional[str]:
    """Parse a multi-choice response to extract the choice letter.

    Args:
        response: Raw model response
        answer_choices: List of answer choices

    Returns:
        Single letter (A/B/C/D/E) or None if parsing fails
    """
    # First, use extract_final_answer() to handle boxed/ANSWER: extraction
    extracted = extract_final_answer(str(response)).strip().upper()

    # Check if extracted answer is a single letter A-D
    if len(extracted) == 1 and extracted in "ABCD":
        return extracted

    # Also check if extracted answer contains a letter (e.g., "A" or "answer is A")
    letter_match = re.search(r"\b([ABCD])\b", extracted)
    if letter_match:
        return letter_match.group(1)

    # Clean response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    all_choices = [chr(ord("A") + i) for i in range(len(answer_choices))]
    index2ans = {choice: ans for choice, ans in zip(all_choices, answer_choices)}

    index_ans = True
    ans_with_brack = False
    candidates = []

    # Look for (A) style answers
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Look for A style answers
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    # Look for A. style answers
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If response is long, try to match the content
    if len(candidates) == 0 and len(response.split()) > 5:
        for choice, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(choice)
                index_ans = False

    # Determine final prediction
    if len(candidates) == 0:
        # Do not guess if no unambiguous choice is found
        return None
    elif len(candidates) > 1:
        # Get the last mentioned choice
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute AI2D-specific metrics.

    Args:
        results: List of result dictionaries with scores

    Returns:
        Dictionary of metrics
    """
    scores = []
    for result in results:
        score_details = result.get("score_details", result)
        scores.append(score_details.get("score", 0.0))

    accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "accuracy": accuracy,
    }
