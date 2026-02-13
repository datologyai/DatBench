"""ChartQAPro scoring.

This is a complex multi-turn QA scorer with relaxed correctness.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple
from .evaluation_utils.erma_utils import extract_final_answer


def _resolve_chartqapro_answers(sample: Dict[str, Any]) -> List[str]:
    """Resolve canonical ChartQAPro answers with legacy fallback."""
    answer_raw = sample.get("all_answers")
    if isinstance(answer_raw, list) and answer_raw:
        return answer_raw

    legacy_answer = sample.get("Answer") or sample.get("ground_truth_answer")
    if isinstance(legacy_answer, list):
        return legacy_answer
    if legacy_answer is not None:
        return [legacy_answer]

    canonical_answer = sample.get("answer")
    if canonical_answer is not None:
        return [canonical_answer]
    return []


def score_sample(sample: Dict[str, Any], model_output: Any) -> Dict[str, Any]:
    """Score a single ChartQAPro sample using the official scoring logic.

    Args:
        sample: Original sample data
        model_output: Model's generated output (string for single-turn, list for multi-turn)

    Returns:
        Dictionary containing scoring results
    """
    # Check if this is a multi-turn sample
    is_multi_turn = sample.get("is_multi_turn", False)
    # Also check if model_output is a list (from multi-turn inference)
    if isinstance(model_output, list):
        is_multi_turn = True

    if is_multi_turn:
        # Multi-turn scoring: score each turn independently
        answer_raw = _resolve_chartqapro_answers(sample)
        year_flags = sample.get("Year") or sample.get("year_flags", [])
        question_type = sample.get("Question Type") or sample.get(
            "question_type", "unknown"
        )

        # Ensure we have lists
        if not isinstance(year_flags, list):
            year_flags = [year_flags]
        if not isinstance(model_output, list):
            model_output = [model_output]

        # Score each turn
        turn_scores = []
        turn_details = []
        num_turns = min(
            len(answer_raw),
            len(model_output),
            len(year_flags) if year_flags else len(answer_raw),
        )

        for turn_idx in range(num_turns):
            target = (
                str(answer_raw[turn_idx]).strip(".").strip("\n")
                if turn_idx < len(answer_raw)
                else ""
            )
            prediction_raw = (
                model_output[turn_idx] if turn_idx < len(model_output) else ""
            )
            prediction = (
                extract_final_answer(str(prediction_raw)).strip(".").strip("\n")
            )

            # Get year flag for this turn (use last one for conversational as per original script)
            turn_year_flags = (
                year_flags[-1:]
                if question_type == "Conversational"
                else (
                    [year_flags[turn_idx]]
                    if turn_idx < len(year_flags)
                    else [year_flags[0]]
                    if year_flags
                    else []
                )
            )

            # Determine if we should use exact match
            always_use_exact_match = question_type in [
                "Fact Checking",
                "Multi Choice",
            ]

            # For Multi Choice, extract letter from prediction (e.g., "b) text" -> "B")
            if question_type == "Multi Choice":
                # Check if this was converted from MCQ to generative
                target_str = str(target).strip().upper()
                is_single_letter = len(target_str) == 1 and target_str in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]
                has_conversion = (
                    sample.get("mcq_conversion") is not None or not is_single_letter
                )
                if not has_conversion:
                    prediction = _extract_choice_letter(prediction)

            # Extract question text for this turn
            all_questions = sample.get("all_questions") or sample.get(
                "question", []
            )
            if isinstance(all_questions, list) and turn_idx < len(all_questions):
                turn_question = all_questions[turn_idx]
            elif isinstance(all_questions, str):
                turn_question = all_questions
            else:
                turn_question = sample.get("question", "")

            # Score this turn
            turn_score = _relaxed_correctness_chartqapro(
                target=target,
                prediction=prediction,
                year_flags=turn_year_flags,
                always_use_exact_match=always_use_exact_match,
                question_text=turn_question,
            )

            turn_scores.append(turn_score)
            turn_details.append(
                {
                    "turn": turn_idx + 1,
                    "score": turn_score,
                    "ground_truth": target,
                    "prediction": prediction,
                }
            )

        # Overall score is average of all turns
        overall_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

        return {
            "score": overall_score,
            "ground_truth": answer_raw,  # List of all answers
            "model_output": model_output,  # List of all responses
            "pred_answer": [
                extract_final_answer(str(r)).strip(".").strip("\n")
                for r in model_output
            ],
            "question_type": question_type,
            "is_multi_turn": True,
            "turn_scores": turn_scores,
            "turn_details": turn_details,
        }
    else:
        # Single-turn scoring: original logic
        answer_raw = _resolve_chartqapro_answers(sample)
        # Answer might be a list, extract last element like the official script
        if isinstance(answer_raw, list) and len(answer_raw) > 0:
            target = str(answer_raw[-1]).strip(".").strip("\n")
        else:
            target = str(answer_raw).strip(".").strip("\n") if answer_raw else ""

        question_type = sample.get("Question Type") or sample.get(
            "question_type", "unknown"
        )
        year_flags = sample.get("Year") or sample.get("year_flags", [])

        # Extract final answer from model output
        prediction = extract_final_answer(str(model_output)).strip(".").strip("\n")

        # Handle Conversational questions - use last Year flag
        if question_type == "Conversational":
            year_flags = (
                year_flags[-1:] if isinstance(year_flags, list) else [year_flags]
            )

        # Determine if we should use exact match
        always_use_exact_match = question_type in ["Fact Checking", "Multi Choice"]

        # For Multi Choice, extract letter from prediction
        if question_type == "Multi Choice":
            target_str = str(target).strip().upper()
            is_single_letter = len(target_str) == 1 and target_str in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]
            has_conversion = (
                sample.get("mcq_conversion") is not None or not is_single_letter
            )
            if not has_conversion:
                prediction = _extract_choice_letter(prediction)

        # Extract question text
        question_text = sample.get("question", "")

        # Calculate score using relaxed correctness
        score = _relaxed_correctness_chartqapro(
            target=target,
            prediction=prediction,
            year_flags=year_flags,
            always_use_exact_match=always_use_exact_match,
            question_text=question_text,
        )

        return {
            "score": score,
            "ground_truth": target,
            "model_output": str(model_output),
            "pred_answer": prediction,
            "question_type": question_type,
            "is_multi_turn": False,
        }


def _fix_list_format(item: str) -> Any:
    """Standardize string representations of lists."""
    if not isinstance(item, str):
        return item

    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item

    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def _parse_to_list(text: str) -> Optional[List[str]]:
    """Parses text to a list of strings if possible."""
    if not isinstance(text, str):
        return None

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None

    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def _normalize_answer(answer: str) -> str:
    """Normalize answer text for comparison (matching ChartQA logic)."""
    if not answer:
        return ""

    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()

    # Remove punctuation and extra whitespace (keep alnum + whitespace)
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()


def _extract_choice_letter(text: str) -> str:
    """Extract the choice letter from model output (e.g., "b) text" -> "B")."""
    text = str(text).strip()
    # Match patterns like "a)", "A)", "a.", "A."
    match = re.match(r"^([a-zA-Z])[).]", text)
    if match:
        return match.group(1).upper()
    # If no match, try to find a single letter at the start
    if len(text) > 0 and text[0].isalpha() and len(text) == 1:
        return text.upper()
    # Return original if no pattern matches
    return text


def _evaluate_single_answer(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    question_text: Optional[str] = None,
) -> float:
    """Evaluates a single target-prediction pair (matching ChartQA logic)."""

    def strip_commas(s: str) -> str:
        return s.replace(",", "") if isinstance(s, str) else s

    def to_float_and_pct_flag(text: str):
        text = (text or "").strip()
        text_no_commas = strip_commas(text)
        is_percent = text_no_commas.endswith("%")
        core = text_no_commas.rstrip("%") if is_percent else text_no_commas
        try:
            val = float(core)
            return val, is_percent
        except ValueError:
            return None, False

    def parse_ratio(text: str) -> Optional[float]:
        text = (text or "").strip()
        text_no_commas = strip_commas(text)
        if "/" in text_no_commas:
            parts = text_no_commas.split("/")
            if len(parts) == 2:
                try:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if abs(denominator - 100.0) < 1e-6:
                        return numerator / 100.0
                except (ValueError, ZeroDivisionError):
                    pass
        return None

    # Check if question mentions "percentage"
    is_percentage_question = (
        question_text is not None and "percentage" in question_text.lower()
    )

    # Extract numeric values and percent flags
    t_f, t_is_pct = to_float_and_pct_flag(target)
    p_f, p_is_pct = to_float_and_pct_flag(prediction)

    # For percentage questions, normalize x% ↔ x/100
    if is_percentage_question and t_f is not None and p_f is not None:
        t_ratio = parse_ratio(target)
        p_ratio = parse_ratio(prediction)

        if t_ratio is not None:
            t_f = t_ratio * 100.0
            t_is_pct = True
        if p_ratio is not None:
            p_f = p_ratio * 100.0
            p_is_pct = True

        # Handle decimal ratios (0.6) as percentages
        if not t_is_pct and 0 <= t_f <= 1.5:
            t_f = t_f * 100.0
            t_is_pct = True
        if not p_is_pct and 0 <= p_f <= 1.5:
            p_f = p_f * 100.0
            p_is_pct = True

    if t_f is not None and p_f is not None:
        # Harmonize percent conventions
        if t_is_pct and not p_is_pct:
            t_f_eff = t_f
            p_f_eff = p_f
        elif p_is_pct and not t_is_pct:
            t_f_eff = t_f
            p_f_eff = p_f
        else:
            t_f_eff = t_f
            p_f_eff = p_f

        # If one side looks like a ratio (<=1.5) and the other like percent (>=2), align by *100
        if t_is_pct != p_is_pct:
            if (
                not p_is_pct
                and t_f_eff is not None
                and t_f_eff >= 2
                and p_f_eff is not None
                and 0 <= p_f_eff <= 1.5
            ):
                p_f_eff = p_f_eff * 100.0
            if (
                not t_is_pct
                and p_f_eff is not None
                and p_f_eff >= 2
                and t_f_eff is not None
                and 0 <= t_f_eff <= 1.5
            ):
                t_f_eff = t_f_eff * 100.0

        if t_f_eff is not None and t_f_eff != 0:
            rel = abs(p_f_eff - t_f_eff) / abs(t_f_eff)
            if rel <= max_relative_change + 1e-10:
                return 1.0
            if abs(t_f_eff) < 1e-6 and abs(p_f_eff - t_f_eff) <= 1e-6:
                return 1.0
        elif t_f_eff == 0.0:
            return 1.0 if p_f_eff == 0.0 else 0.0

    # List pathway
    def parse_list(ans: str) -> List[str]:
        if not isinstance(ans, str):
            return []
        s = ans.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            inner = s[1:-1]
            items = [it.strip() for it in inner.split(",") if it.strip()]
            norm_items = []
            for it in items:
                itn = _normalize_answer(it)
                if itn:
                    norm_items.append(itn)
            return sorted(norm_items)
        return []

    p_list = parse_list(prediction)
    t_list = parse_list(target)
    if p_list and t_list:
        return 1.0 if p_list == t_list else 0.0

    # Non-numeric text
    p_norm = _normalize_answer(prediction)
    t_norm = _normalize_answer(target)
    return 1.0 if p_norm == t_norm else 0.0


def _relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: Optional[List[Any]] = None,
    always_use_exact_match: bool = False,
    question_text: Optional[str] = None,
) -> float:
    """Calculates relaxed correctness between target and prediction."""
    fixed_t = _fix_list_format(target)
    t_list = _parse_to_list(str(fixed_t)) or [str(target)]
    p_list = _parse_to_list(str(prediction)) or [str(prediction)]

    n = len(t_list)

    # Expand year_flags for questions with multiple answers
    if year_flags is not None:
        if not isinstance(year_flags, list):
            year_flags = [year_flags]
        year_flags = [str(flag).upper() for flag in year_flags]
        if len(year_flags) < n:
            year_flags = year_flags * n

    # Evaluate elements
    scores: List[float] = []

    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue

        t_item, p_item = t_list[idx], p_list[idx]
        flag = year_flags[idx] if year_flags and idx < len(year_flags) else "NO"
        flag_cond = True if flag.upper() == "YES" else False

        if flag_cond or always_use_exact_match:
            # Exact match pathway
            t_norm = _normalize_answer(str(t_item))
            p_norm = _normalize_answer(str(p_item))
            scores.append(1.0 if t_norm == p_norm else 0.0)
        else:
            scores.append(
                _evaluate_single_answer(
                    t_item, p_item, max_relative_change, question_text
                )
            )

    return sum(scores) / len(scores) if scores else 0.0


def _extract_choice_letter(text: str) -> str:
    """Extract the choice letter from model output (e.g., "b) text" -> "B")."""
    text = str(text).strip()
    match = re.match(r"^([a-zA-Z])[).]", text)
    if match:
        return match.group(1).upper()
    if len(text) > 0 and text[0].isalpha() and len(text) == 1:
        return text.upper()
    return text


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute ChartQAPro-specific metrics."""
    scores = []
    for result in results:
        score_details = result.get("score_details", result)
        scores.append(score_details.get("score", 0.0))

    accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "accuracy": accuracy,
    }
