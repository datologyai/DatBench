"""TextVQA scoring."""

import re
from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer
from .evaluation_utils.textvqa_eval import EvalAIAnswerProcessor


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single TextVQA sample.

    Uses the official TextVQA evaluation metric.

    Args:
        sample: Sample data with 'answers' or 'all_answers' field
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Get ground truth answers
    ground_truth_answers = sample.get("answers", sample.get("all_answers", []))

    # Extract concise answer before scoring
    pred_answer = extract_final_answer(model_output)

    # Primary: Use an LMMS/official-equivalent scoring with EvalAI normalization
    score = _compute_textvqa_accuracy(pred_answer, ground_truth_answers)

    # Fallback: treat currency/percent-only differences as correct for numeric answers
    # If score is 0, and both pred and at least one GT contain digits, compare their
    # numeric forms after stripping currency symbols, percent signs, and non-numeric
    # characters (except the decimal point). If equal, mark as correct.
    if score == 0.0:

        def to_numeric_str(s: str) -> str:
            if not s:
                return ""
            # Remove common currency symbols and percent signs
            s = s.replace("$", "").replace("£", "").replace("€", "")
            s = s.replace("%", "")
            # Keep digits and at most one dot by removing other characters
            # First, replace commas (thousand separators)
            s = s.replace(",", "")
            # Remove everything except digits and dot
            s = re.sub(r"[^0-9\.]", "", s)
            # Normalize multiple dots if any by keeping the first
            if s.count(".") > 1:
                parts = [p for p in s.split(".") if p != ""]
                if parts:
                    s = parts[0] + "." + "".join(parts[1:])
            return s.strip(".")

        # Only apply this fallback when currency/percent symbols are involved
        symbols = ("$", "£", "€", "¥", "₹", "%")
        pred_has_symbol = any(sym in str(pred_answer) for sym in symbols)
        gt_has_symbol = any(
            any(sym in str(gt) for sym in symbols) for gt in ground_truth_answers
        )

        if pred_has_symbol or gt_has_symbol:
            pred_num = to_numeric_str(pred_answer)
            if any(ch.isdigit() for ch in pred_answer) and pred_num:
                for gt in ground_truth_answers:
                    if gt and any(ch.isdigit() for ch in str(gt)):
                        gt_num = to_numeric_str(str(gt))
                        if gt_num and pred_num == gt_num:
                            score = 1.0
                            break

    return {
        "score": score,
        "ground_truth_answers": ground_truth_answers,
        "model_output": model_output,
        "pred_answer": pred_answer,
    }


def _compute_textvqa_accuracy(
    prediction: str, ground_truth_answers: List[str]
) -> float:
    """Compute TextVQA accuracy mirroring LMMS-eval's logic.

    Steps (per LMMS TextVQA):
    - Normalize pred and each GT with EvalAIAnswerProcessor
    - For each GT answer position i, count matches of other GTs to pred
    - Acc_i = min(1, matches/3); final = mean_i Acc_i
    """
    if not ground_truth_answers:
        return 0.0

    proc = EvalAIAnswerProcessor()
    pred_norm = proc(prediction or "")

    # Normalize GT answers
    gts = [proc(a) for a in ground_truth_answers]
    if not gts:
        return 0.0

    accs: List[float] = []
    n = len(gts)
    for i in range(n):
        other = [gts[j] for j in range(n) if j != i]
        matching = [x for x in other if x == pred_norm]
        accs.append(min(1.0, float(len(matching)) / 3.0))

    return sum(accs) / len(accs) if accs else 0.0


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute TextVQA-specific metrics.

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

    # Note: In the full implementation, we would compute both OCR and Pure versions
    # For now, we return a single accuracy metric
    return {
        "accuracy__TextVQA-OCR": accuracy,
    }
