"""MMMUPro scoring."""

import ast
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score an MMMU Pro sample.

    Args:
        sample: Original sample data or metadata dict
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Handle both original sample dict and metadata dict
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer", "")
    options = sample.get("options", [])
    all_choices = sample.get("all_choices", [])
    index2ans = sample.get("index2ans", {})

    # If options not parsed yet, build them
    if not all_choices or not index2ans:
        if isinstance(options, str):
            options = ast.literal_eval(options)
        index2ans, all_choices = _get_multi_choice_info(options)

    # Parse multiple choice response
    parsed_response = _parse_multi_choice_response(
        model_output, all_choices, index2ans
    )

    # Evaluate
    is_correct = _eval_multi_choice(ground_truth, parsed_response)

    return {
        "score": 1.0 if is_correct else 0.0,
        "extracted_output": parsed_response,
        "pred_answer": parsed_response,
        "is_correct": is_correct,
    }


def _get_multi_choice_info(
    options: List[str]
) -> Tuple[Dict[str, str], List[str]]:
    """Build index2ans mapping and all_choices list from options.

    Args:
        options: List of option strings

    Returns:
        Tuple of (index2ans dict, all_choices list)
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}

    for i, option in enumerate(options):
        choice_letter = chr(ord(start_chr) + i)
        index2ans[choice_letter] = str(option)
        all_choices.append(choice_letter)

    return index2ans, all_choices


def _parse_multi_choice_response(
    response: str, all_choices: List[str], index2ans: Dict[str, str]
) -> Optional[str]:
    """Parse the prediction from the generated response.

    Based on MMMU's parse_multi_choice_response function.

    Args:
        response: Model's generated response
        all_choices: List of choice letters (A-J)
        index2ans: Mapping from letter to option text

    Returns:
        Predicted choice letter or None if not found
    """
    if not all_choices:
        return None

    # First, try to extract from boxed format using extract_final_answer
    extracted = extract_final_answer(str(response)).strip().upper()

    # Check if extracted answer is a single letter A-J
    if len(extracted) == 1 and extracted in all_choices:
        return extracted

    # Also check if extracted answer contains a letter
    letter_match = re.search(rf"\b([{'|'.join(all_choices)}])\b", extracted)
    if letter_match:
        return letter_match.group(1)

    # Clean response
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        response = ""

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
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

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
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def _eval_multi_choice(gold_i: str, pred_i: Optional[str]) -> bool:
    """Evaluate a multiple choice instance.

    Args:
        gold_i: Ground truth answer (letter or list of letters)
        pred_i: Predicted answer (letter)

    Returns:
        True if correct, False otherwise
    """
    if pred_i is None:
        return False

    if isinstance(gold_i, list):
        for answer in gold_i:
            if str(answer).upper() == str(pred_i).upper():
                return True
    else:
        if str(gold_i).upper() == str(pred_i).upper():
            return True
    return False


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute MMMU Pro-specific metrics.

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
