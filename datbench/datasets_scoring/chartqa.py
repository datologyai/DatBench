"""ChartQA scoring."""

import re
from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single ChartQA sample using relaxed correctness (LMMS spec).

    - Extract final short answer from ERMA/CoT output
    - If both GT and pred parse as numbers (percent-aware), accept within 5% relative error
    - Otherwise, require case-insensitive exact match
    """
    # Handle both original sample dict and metadata dict from evaluate script
    answer = sample.get("answer") or sample.get("ground_truth_answer")
    all_ground_truth_answers = sample.get("all_answers") or sample.get(
        "all_ground_truth_answers", [answer] if answer else []
    )

    # Ensure we have a list of answers
    if not isinstance(all_ground_truth_answers, list):
        all_ground_truth_answers = [all_ground_truth_answers]

    # Remove empty answers
    all_ground_truth_answers = [ans for ans in all_ground_truth_answers if ans]

    score = 0.0
    short_pred = extract_final_answer(model_output)
    if all_ground_truth_answers:
        # Evaluate relaxed correctness against any GT
        for gt_answer in all_ground_truth_answers:
            if _relaxed_correctness(short_pred, gt_answer, sample):
                score = 1.0
                break

    return {
        "score": score,
        "ground_truth": answer,
        "all_ground_truth_answers": all_ground_truth_answers,
        "model_output": model_output,
        "pred_answer": short_pred,
    }


def _normalize_answer(answer: str) -> str:
    """Normalize answer text for comparison."""
    if not answer:
        return ""

    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()

    # Remove punctuation and extra whitespace (keep alnum + whitespace)
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()


def _relaxed_correctness(
    prediction: str,
    target: str,
    sample: Dict[str, Any] | None = None,
    max_relative_change: float = 0.05,
) -> bool:
    """ChartQA relaxed correctness with pragmatic normalization.

    Rules:
    - Numeric: remove thousands separators; handle percent forms; 5% relative tolerance.
    - Non-numeric text: strip simple punctuation and normalize whitespace (case-insensitive).
    - Lists: if both look like bracketed lists, compare as order-insensitive sets of items.
    """

    def strip_commas(s: str) -> str:
        return s.replace(",", "") if isinstance(s, str) else s

    def to_float_and_flags(text: str):
        try:
            t = (text or "").strip()
            t_no_commas = strip_commas(t)
            is_percent = t_no_commas.endswith("%")
            core = t_no_commas.rstrip("%") if is_percent else t_no_commas
            # Accept forms like "16.9" or "16.9%"
            val = float(core)
            return val, is_percent
        except Exception:
            return None, False

    # Numeric pathway with percent + thousands normalization
    pv, p_is_pct = to_float_and_flags(prediction)
    gv, g_is_pct = to_float_and_flags(target)

    if pv is not None and gv is not None:
        # Harmonize percent conventions:
        # - If exactly one side has %, interpret the other in the same unit when plausible.
        #   If the non-% side is in [0, 1.5], assume it's a ratio → scale by 100.
        #   Else assume it's already a percent value (e.g., 24 means 24%).
        if p_is_pct and not g_is_pct:
            # Bring pred to same scale as target (no '%'): interpret pred '%' as plain value
            pv_eff = pv  # e.g., 24% -> 24
            gv_eff = gv if gv is not None else None
        elif g_is_pct and not p_is_pct:
            pv_eff = pv
            gv_eff = gv
        else:
            pv_eff = pv
            gv_eff = gv

        # If one side looks like a ratio (<=1.5) and the other like percent (>=2), align by *100
        if p_is_pct != g_is_pct:
            if (
                not p_is_pct
                and gv_eff is not None
                and gv_eff >= 2
                and pv_eff is not None
                and 0 <= pv_eff <= 1.5
            ):
                pv_eff = pv_eff * 100.0
            if (
                not g_is_pct
                and pv_eff is not None
                and pv_eff >= 2
                and gv_eff is not None
                and 0 <= gv_eff <= 1.5
            ):
                gv_eff = gv_eff * 100.0

        if gv_eff is not None and gv_eff != 0:
            rel = abs(pv_eff - gv_eff) / abs(gv_eff)
            if rel <= max_relative_change:
                return True
            # Also allow absolute small epsilon for tiny targets
            if abs(gv_eff) < 1e-6 and abs(pv_eff - gv_eff) <= 1e-6:
                return True

    # List (order-insensitive) pathway: detect [a, b, ...]
    def parse_list(ans: str) -> List[str]:
        if not isinstance(ans, str):
            return []
        s = ans.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            inner = s[1:-1]
            items = [it.strip() for it in inner.split(",") if it.strip()]
            # normalize items: lowercase, strip punctuation and condense whitespace
            norm_items = []
            for it in items:
                itn = _normalize_answer(it)
                if itn:
                    norm_items.append(itn)
            return sorted(norm_items)
        return []

    p_list = parse_list(prediction)
    g_list = parse_list(target)
    if p_list and g_list:
        return p_list == g_list

    # Non-numeric text: normalized case-insensitive equality
    p_norm = _normalize_answer(prediction)
    g_norm = _normalize_answer(target)
    return p_norm == g_norm


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute ChartQA-specific metrics.

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
        "accuracy__ChartQA-Relaxed": accuracy,
    }
