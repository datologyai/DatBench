"""RealWorldQA scoring."""

import re
import string
from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score RealWorldQA per LMMS-eval: exact match after light normalization.

    Pipeline:
    1) Extract final answer text from ERMA/CoT (prefer last 'Answer:').
    2) Apply filters: number words→digits; MC regex mapping question choices→letter.
    3) Normalize pred and gt (lower/strip, rstrip('.')) and exact match.
    """
    ground_truth = sample.get("ground_truth_answer") or sample.get("answer") or ""
    question_text = (
        sample.get("question_text")
        or sample.get("question_raw")
        or sample.get("question")
        or ""
    )

    # 1) Extract concise final answer from ERMA/CoT
    final_pred = extract_final_answer(model_output)

    # 2a) Number words to digits (zero..ten)
    num_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    pred_filtered = num_map.get(final_pred.lower().strip(), final_pred)

    # 2b) Multiple-choice mapping (robust):
    #  - Accept standalone letter lines (A/B/C/D/E)
    #  - Token-boundary letter from model_output 'Answer:' or short lines
    #  - Fuzzy match choice text (>=0.8 token overlap) if letters aren't parsed
    # Parse choices from question text (e.g., lines like 'A. ...', 'B. ...')
    choices_map = _parse_choices_from_question(
        question_text or ""
    )  # letter -> normalized text
    if not choices_map:
        # Fall back to explicit options in metadata/sample
        options_list = (
            sample.get("options")
            or sample.get("candidate_answers")
            or sample.get("answer_choices")
            or []
        )
        if isinstance(options_list, (list, tuple)):
            choices_map = {
                chr(ord("a") + idx): _normalize_answer(opt)
                for idx, opt in enumerate(options_list)
            }
    if choices_map:
        # Try explicit letter in final answer
        letter = _normalize_letter(final_pred)
        if letter and letter.lower() in choices_map:
            pred_filtered = letter
        else:
            # Try extracting letter from model output (Answer: X or short lines)
            letter2 = _extract_predicted_letter(model_output)
            if letter2 and letter2.lower() in choices_map:
                pred_filtered = letter2
            else:
                # Fuzzy match choice text with strict threshold
                def _norm(s: str) -> str:
                    s = (s or "").lower()
                    s = re.sub(r"[^a-z0-9\s]", " ", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    return s

                predN = _norm(pred_filtered)
                if predN:
                    best = None
                    best_score = 0.0
                    for lett, txt in choices_map.items():
                        ansN = _norm(txt)
                        if not ansN:
                            continue
                        set_a = set(ansN.split())
                        set_b = set(predN.split())
                        if not set_a:
                            continue
                        jacc = len(set_a & set_b) / float(len(set_a))
                        if jacc > best_score:
                            best_score = jacc
                            best = lett
                    if best is not None and best_score >= 0.8:
                        pred_filtered = best.upper()

    # 3) Normalize and exact match
    def norm_pred(s: str) -> str:
        # Spec: pred lower/strip/rstrip('.')
        return (s or "").lower().strip().rstrip(".")

    def norm_gt(s: str) -> str:
        # Spec: gt lower/strip
        return (s or "").lower().strip()

    pred_norm = norm_pred(pred_filtered)
    gt_norm = norm_gt(str(ground_truth))
    score = 1.0 if pred_norm == gt_norm and pred_norm != "" else 0.0

    return {
        "score": score,
        "ground_truth": ground_truth,
        "pred_answer": pred_filtered,
        "model_output": model_output,
    }


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase
    answer = answer.lower()

    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    answer = re.sub(r"\s+", " ", answer).strip()

    return answer


def _strip_think(text: str) -> str:
    """Remove <think>...</think> sections from the text."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)


def _normalize_letter(s: str):
    s = (s or "").strip().upper()
    return s if s in {"A", "B", "C", "D", "E"} else None


def _extract_predicted_letter(model_output: str):
    if not isinstance(model_output, str):
        return None
    t = _strip_think(model_output)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    # Prefer last 'Answer:' line
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"(?i)^\s*answer\s*:\s*(.+)$", lines[i].strip())
        if m:
            ans = m.group(1).strip()
            m1 = re.search(r"\b([A-Ea-e])\b", ans)
            if m1:
                return m1.group(1).upper()
            m2 = re.search(r"(?:^|\s)[\(\[]?([A-Ea-e])[\)\]\.!?,]?(?:\s|$)", ans)
            if m2:
                return m2.group(1).upper()
            break
    # Fallback: last short line (<= 10 tokens)
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if len(ln.split()) <= 10:
            m1 = re.search(r"\b([A-Ea-e])\b", ln)
            if m1:
                return m1.group(1).upper()
            m2 = re.search(r"(?:^|\s)[\(\[]?([A-Ea-e])[\)\]\.!?,]?(?:\s|$)", ln)
            if m2:
                return m2.group(1).upper()
            break
    return None


def _parse_choices_from_question(question_text: str):
    """Parse multiple-choice options from question text of the form 'A. ...', 'B. ...'. Returns map letter->normalized text."""
    if not isinstance(question_text, str) or not question_text:
        return {}
    choices = {}
    for line in question_text.splitlines():
        m = re.match(r"\s*([A-Ea-e])\s*\.\s*(.+)$", line)
        if m:
            letter = m.group(1).lower()
            txt = _normalize_answer(m.group(2))
            choices[letter] = txt
    return choices


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute RealWorldQA-specific metrics.

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
        "accuracy__RealWorldQA-Accuracy": accuracy,
    }
