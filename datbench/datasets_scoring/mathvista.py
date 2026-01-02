"""MathVista scoring."""

import re
import numpy as np
from typing import Any, Dict, List, Optional
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single MathVista sample.

    Uses different scoring methods based on question type.

    Args:
        sample: Sample data or metadata dict containing ground truth
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Handle both original sample dict and metadata dict
    question_type = sample.get("question_type", "free_form")
    ground_truth = sample.get("ground_truth_answer") or sample.get("answer")

    score = 0.0
    parsed_response = None

    if question_type == "multi_choice":
        # Parse multiple choice response
        all_choices = sample.get("all_choices", [])
        if all_choices:
            # Build index2ans mapping from choices
            choices = sample.get("choices", [])
            index2ans = {}
            for i, choice in enumerate(choices):
                if i < len(all_choices):
                    index2ans[all_choices[i]] = choice

            # Use extract_final_answer() to handle boxed/ANSWER: extraction first
            extracted = extract_final_answer(str(model_output)).strip().upper()
            # Check if extracted answer is a single letter A-E
            if len(extracted) == 1 and extracted in all_choices:
                parsed_response = extracted
            else:
                # Extract letter from extracted answer or fall back to full parsing
                letter_match = re.search(r"\b([A-E])\b", extracted)
                if letter_match and letter_match.group(1) in all_choices:
                    parsed_response = letter_match.group(1)
                else:
                    parsed_response = _parse_multi_choice_response(
                        model_output, all_choices, index2ans
                    )
            score = (
                1.0
                if _eval_multi_choice(ground_truth, parsed_response)
                else 0.0
            )
        else:
            # Fallback to simple comparison
            score = 1.0 if ground_truth in model_output else 0.0
    else:
        # Free-form question - parse open response
        # Use extract_final_answer() to handle boxed/ANSWER: extraction first
        extracted = extract_final_answer(str(model_output)).strip()
        if extracted:
            parsed_response = _normalize_str(extracted)
        else:
            parsed_response = _parse_open_response(model_output)

        # Fallback: if no parse and this appears numeric, extract last numeric token
        def _looks_numeric(s: str) -> bool:
            if not isinstance(s, str):
                s = str(s)
            return bool(re.search(r"[\d%$]", s))

        if parsed_response is None and (
            _looks_numeric(ground_truth) or _looks_numeric(model_output)
        ):
            nums = re.findall(
                r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?", str(model_output)
            )
            if nums:
                parsed_response = _normalize_str(nums[-1].strip())

        score = 1.0 if _eval_open(ground_truth, parsed_response) else 0.0

    return {
        "score": score,
        "ground_truth": ground_truth,
        "question_type": question_type,
        "parsed_response": parsed_response,
        "model_output": model_output,
    }


def _parse_multi_choice_response(
    response: str, all_choices: List[str], index2ans: Dict[str, str]
) -> str:
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    # Prefer explicit final answer line if present (ERMA/CoT style)
    try:
        import re as _re

        lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    except Exception:
        lines = []
    if lines:
        pat = _re.compile(
            r"^(?:final\s+answer|answer)\s*:\s*\\+boxed\{\s*([A-E])\s*\}\s*$",
            _re.IGNORECASE,
        )
        for ln in reversed(lines):
            m = pat.match(ln)
            if m:
                return m.group(1).upper()
        pat_letter = _re.compile(
            r"^(?:final\s+answer|answer)\s*:\s*([A-E])\b", _re.IGNORECASE
        )
        for ln in reversed(lines):
            m = pat_letter.match(ln)
            if m:
                return m.group(1).upper()

    boxed_inline = re.search(r"\\+boxed\{\s*([A-E])\s*\}", response)
    if boxed_inline:
        return boxed_inline.group(1).upper()

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    # Accept a standalone letter line
    lone = response.strip().upper()
    if lone in all_choices:
        return lone

    # Token-boundary single-letter match (last occurrence)
    m = re.findall(r"\b([A-E])\b", response, flags=re.IGNORECASE)
    if m:
        for ch in reversed(m):
            chU = ch.upper()
            if chU in all_choices:
                return chU

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if (
        len(candidates) == 0
    ):  # still not get answer, do not guess; try fuzzy option text

        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        respN = _norm(response)
        if index2ans:
            best = None
            best_score = 0.0
            for idx, ans in index2ans.items():
                ansN = _norm(ans)
                if not ansN:
                    continue
                set_a = set(ansN.split())
                set_b = set(respN.split())
                if not set_a:
                    continue
                # overlap relative to answer tokens
                jacc = len(set_a & set_b) / float(len(set_a))
                if jacc > best_score:
                    best_score = jacc
                    best = idx
            if best is not None and best_score >= 0.8 and best in all_choices:
                return best
        return None
    elif len(candidates) > 1:
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


def _parse_open_response(response: str) -> List[str]:
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
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
        key_responses = []
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
                # and it's not trivial
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
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # Prefer the concise final answer when present (ERMA/CoT aware)
    # extract_final_answer() now handles boxed extraction, so no need for redundant check
    short = extract_final_answer(response)
    key_responses = [short] if short else get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(_normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def _extract_numbers(string: str) -> List[str]:
    """Extract all forms of numbers from a string with regex."""
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
    """Check if the given string is a number."""
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(string: str) -> List[str]:
    """Normalize the str to lower case and make them float numbers if possible."""
    string = string.strip()
    is_number = _check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def _eval_multi_choice(gold_i: str, pred_i: str) -> bool:
    """Evaluate a multiple choice instance."""
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def _eval_open(gold_i: str, pred_i: List[str]) -> bool:
    """Evaluate an open question instance."""
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(gold_i)

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
    """Compute MathVista-specific metrics.

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
