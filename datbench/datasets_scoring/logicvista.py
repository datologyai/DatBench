"""LogicVista scoring."""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Set
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a LogicVista sample.

    For multiple correct answers (e.g., "A, C"), the model must get
    ALL correct answers and NO extra options to be scored as correct.

    Args:
        sample: Original sample data
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Parse ground truth as set of correct letters
    ground_truth = sample.get("answer", "")
    correct_answer_letters_list = sample.get("correct_answer_letters")
    if correct_answer_letters_list is None:
        # Fallback: parse from ground truth string
        correct_answer_letters = _parse_answer_letters(ground_truth)
    else:
        # Convert list back to set for comparison
        correct_answer_letters = set(correct_answer_letters_list)

    # Parse model response to extract all selected letters
    predicted_letters = _parse_answer_letters(model_output)

    # Also try parsing using the existing method for single-letter responses
    # We need to extract choices from question text for the parser
    question_text = (
        sample.get("question_raw")
        or sample.get("question")
        or sample.get("question_text")
        or ""
    )
    choices_map = _parse_choices_from_question(question_text)
    answer_choices = (
        [choices_map[letter] for letter in sorted(choices_map.keys())]
        if choices_map
        else (
            sample.get("options")
            or sample.get("candidate_answers")
            or []
        )
    )

    parsed_response = None
    if answer_choices:
        parsed_response = _parse_multi_choice_response(
            model_output, answer_choices
        )
        if parsed_response:
            predicted_letters.add(parsed_response)

    # Check for exact match: all correct answers and no extras
    is_correct = predicted_letters == correct_answer_letters

    # For backward compatibility, also compute predicted_index
    predicted_index = None
    if len(predicted_letters) == 1:
        letter = list(predicted_letters)[0]
        if letter in "ABCDEFGH":
            predicted_index = ord(letter) - ord("A")

    # Format extracted output as string (e.g., "A, C" or "A")
    if predicted_letters:
        extracted_output = ", ".join(sorted(predicted_letters))
    else:
        extracted_output = (
            parsed_response if parsed_response else None
        )  # Fallback to single letter if available

    return {
        "score": 1.0 if is_correct else 0.0,
        "extracted_output": extracted_output,
        "predicted_index": predicted_index,
        "is_correct": is_correct,
        "pred_answer": extracted_output,
    }


def _parse_answer_letters(answer_str: str) -> Set[str]:
    """Parse answer string to extract all valid choice letters.

    Uses extract_final_answer() as wrapper to handle boxed/ANSWER: extraction,
    then extracts letters from the result.
    Handles formats like "A, C", "A,C", "A C", "AC", etc.
    Supports choices A through H.

    Args:
        answer_str: Answer string that may contain one or more letters

    Returns:
        Set of uppercase letters (A-H) found in the answer
    """
    if not answer_str:
        return set()

    # First, use extract_final_answer() to handle boxed/ANSWER: extraction
    extracted = extract_final_answer(answer_str).strip()

    # Extract letters from the extracted answer
    letters = set(re.findall(r"[A-H]", extracted.upper()))
    if letters:
        return letters

    # Fallback: look for explicit "Answer:" or "Final Answer:" patterns
    answer_patterns = [
        r"(?:answer|final\s+answer)\s*:\s*([A-H,\s]+)",
        r"answer\s+is\s+([A-H,\s]+)",
        r"correct\s+answer\s+is\s+([A-H,\s]+)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, answer_str, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            letters = set(re.findall(r"[A-H]", answer_part.upper()))
            if letters:
                return letters

    # Last resort: extract all letters A-H from the entire string
    # But be more conservative - only if the string is relatively short
    # or if we find a clear pattern
    normalized = answer_str.upper().replace(" ", "").replace(",", "")
    all_letters = set(re.findall(r"[A-H]", normalized))

    # If we found too many letters (likely from question text), return empty
    # Only return if we have a reasonable number (1-4 letters)
    if len(all_letters) <= 4 and len(all_letters) > 0:
        return all_letters

    return set()


def _parse_multi_choice_response(
    response: str, answer_choices: List[str]
) -> Optional[str]:
    """Parse the prediction from the generated response.

    Based on AI2D's parse_multi_choice_response function.
    Supports up to 8 choices (A-H).

    Args:
        response: Model's generated response
        answer_choices: List of answer choice texts

    Returns:
        Extracted letter choice (A, B, C, ..., H) or None if not found
        Note: This returns a single letter for backward compatibility.
        For multiple letters, use _parse_answer_letters() instead.
    """
    # Prefer explicit final answer if present: parse last line starting with ANSWER/Final Answer
    lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if lines:
        # Try to match single letter first
        pat = re.compile(
            r"^(?:final\s+answer|answer)\s*:\s*([A-H])\b", re.IGNORECASE
        )
        for ln in reversed(lines):
            m = pat.match(ln)
            if m:
                return m.group(1).upper()

    # Clean response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    # Support up to 8 choices (A-H)
    max_choices = min(len(answer_choices), 8)
    all_choices = [chr(ord("A") + i) for i in range(max_choices)]
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


def _parse_choices_from_question(question_text: str) -> Dict[str, str]:
    """Parse multiple choice options from question text.

    Extracts choices formatted as "A. ...", "B. ...", etc.
    Supports choices A through H (8 options).

    Args:
        question_text: Question text containing options

    Returns:
        Dictionary mapping letter (A, B, C, ..., H) to choice text
    """
    choices_map = {}
    # Pattern to match "A. text", "B. text", etc. (supports A-H)
    pattern = re.compile(r"([A-H])\.\s+([^\n]+)")
    matches = pattern.findall(question_text)

    for letter, choice_text in matches:
        choices_map[letter] = choice_text.strip()

    return choices_map


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute LogicVista-specific metrics.

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
