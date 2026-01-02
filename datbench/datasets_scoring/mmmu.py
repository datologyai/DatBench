"""MMMU scoring."""

import re
import numpy as np
from typing import Any, Dict, List, Optional
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score an MMMU sample.

    Args:
        sample: Original sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    question_type = sample.get("question_type", "multiple-choice")

    if question_type == "multiple-choice":
        # Parse multiple choice response
        parsed_response = _parse_multi_choice_response(
            model_output, sample.get("all_choices", []), sample.get("index2ans", {})
        )

        # Evaluate
        is_correct = _eval_multi_choice(sample.get("answer", ""), parsed_response)

        return {
            "score": 1.0 if is_correct else 0.0,
            "extracted_output": parsed_response,
            "pred_answer": parsed_response,
            "is_correct": is_correct,
        }
    else:
        # Open-ended questions
        parsed_responses = _parse_open_response(model_output)

        # Evaluate
        is_correct = _eval_open(sample.get("answer", ""), parsed_responses)

        # For audit, expose the concise final answer if available
        pred_short = extract_final_answer(model_output)
        return {
            "score": 1.0 if is_correct else 0.0,
            "extracted_output": parsed_responses,
            "pred_answer": pred_short
            or (parsed_responses[0] if parsed_responses else None),
            "is_correct": is_correct,
        }


def _parse_multi_choice_response(
    response: str, all_choices: List[str], index2ans: Dict[str, str]
) -> str:
    """Parse the prediction from the generated response.

    Based on MMMU's parse_multi_choice_response function.
    """
    # Prefer explicit final answer line if present (ERMA/CoT style)
    try:
        import re as _re

        lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    except Exception:
        lines = []
    if lines:
        pat = _re.compile(
            r"^(?:final\s+answer|answer)\s*:\s*([A-E])\b", _re.IGNORECASE
        )
        for ln in reversed(lines):
            m = pat.match(ln)
            if m:
                return m.group(1).upper()
    # Clean response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

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


def _parse_open_response(response: str) -> List[str]:
    """Parse open-ended response.

    Based on MMMU's parse_open_response function.
    """

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]

        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])

            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[
                                -1
                            ].strip()

            if shortest_key_response:
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)

        if len(key_responses) == 0:
            return [response]
        return key_responses

    # Prefer concise ERMA/CoT final answer if available
    short = extract_final_answer(response)
    key_responses = [short] if short else get_key_subresponses(response)
    pred_list = key_responses.copy()

    # Extract numbers
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    # Normalize strings
    tmp_pred_list = []
    for pred in pred_list:
        tmp_pred_list.extend(_normalize_str(pred))
    pred_list = tmp_pred_list

    # Remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def _extract_numbers(string: str) -> List[str]:
    """Extract all forms of numbers from a string with regex.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def _check_is_number(string: str) -> bool:
    """Check if the given string is a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float_val = float(string.replace(",", ""))
        # Even if it converts to inf, it's still technically a number
        return True
    except (ValueError, OverflowError):
        return False


def _normalize_str(string: str) -> List[str]:
    """Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # if number, numerize it.
    string = string.strip()

    is_number = _check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        try:
            float_val = float(string)
            # Check if the value is infinite or too large
            if (
                float_val == float("inf")
                or float_val == float("-inf")
                or abs(float_val) > 1e308
            ):
                # Return the original string for very large numbers
                return [string.lower()]
            # leave 2 decimal
            string = round(float_val, 2)
            return [string]
        except (ValueError, OverflowError):
            # If conversion fails, treat as string
            return [string.lower()]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def _eval_multi_choice(gold_i: str, pred_i: str) -> bool:
    """Evaluate a multiple choice instance."""
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if str(gold_i) == str(pred_i):
            correct = True
    return correct


def _eval_open(gold_i: Any, pred_i: List[str]) -> bool:
    """Evaluate an open question instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(str(gold_i))
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(
            pred, str
        ):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute MMMU-specific metrics.

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

    # Could add subject-wise accuracy here if we track subjects

    return {
        "accuracy": accuracy,
    }
