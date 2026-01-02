"""MathVision scoring utilities.

Mathematical expression parsing and comparison for MathVision dataset.
Adapted from lmms-eval MathVision task.
"""

from __future__ import annotations

import re
from typing import List, Dict

from latex2sympy2 import latex2sympy

# Helpers from lmms-eval eval_utils.py


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_tuple(s: str):
    sl = s[1:-1].split(",")
    try:
        if s[0] == "(" and s[-1] == ")" and len(sl) > 1:
            s_val = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                    for sub in sl
                ]
            )
            return f"({s_val})"
        elif s[0] == "[" and s[-1] == "]" and len(sl) > 1:
            s_val = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                    for sub in sl
                ]
            )
            return f"[{s_val}]"
    except Exception:
        return s
    return s


def is_equal(asw: str, gt_asw: str) -> bool:
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if asw.replace(" ", "") == "" or gt_asw.replace(" ", "") == "":
        return False
    if gt_asw.strip() == asw.strip():
        return True

    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)

    if gt_asw == asw:
        return True

    try:
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
            return True
        return False
    except Exception:
        return False


def _remove_right_units(string: str) -> str:
    splits = string.split("\\text{ ")
    return splits[0]


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if "sqrt" in string:
        string = _fix_sqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def find_math_answer(s: str) -> str:
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")
    pattern = re.compile(r"\\+boxed\{(.*)\}", flags=re.S)
    matches = pattern.findall(s)
    ans = matches[-1] if matches else s
    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]
    ans = ans.split("=")[-1]
    ans = ans.split("\\approx")[-1]
    ans = ans.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    ans = ans.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")
    return _strip_string(ans)


# --------------------------
# Prompt & scoring wrappers
# --------------------------


def build_prompt(question: str, choices: List[str], mc_prompt: str | None = None) -> str:
    options = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
    choice_block = "\nChoices: " + "\n".join(options) if options else ""
    base = 'Please solve the problem step by step and put your answer in one "\\\\boxed{}". (Type two backslashes, e.g., \\\\boxed{A})'
    prompt = base + (question or "")
    if choices:
        prompt = prompt + "\n" + choice_block
        if mc_prompt:
            prompt = prompt + "\n" + mc_prompt
    return prompt


def mathvision_process_results(doc: Dict, results):
    correct_list = []
    for pred in results:
        model_answer = str(pred).strip()
        gt_answer = str(doc.get("answer", ""))
        options = doc.get("options", []) or []
        if options:
            try:
                idx = ord(gt_answer) - ord("A")
                gt_answer_value = options[idx] if 0 <= idx < len(options) else ""
            except Exception:
                gt_answer_value = ""
        else:
            gt_answer_value = ""

        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "\\boxed{" not in model_answer and "\\\\boxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag_cap = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag_cap)[-1].strip()
                if flag_cap in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("\\\\boxed{") > 1:
            model_answer = "\\\\boxed{" + model_answer.split("\\\\boxed{")[-1]
        elif model_answer.count("\\boxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("\\boxed{")[-1]

        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    return {
        "mathvision_standard_eval": {
            "response": results,
            "scores": correct_list,
        },
    }


def mathvision_aggregate_results_eval(results):
    total = len(results)
    correct = sum(1 for idx, result in enumerate(results) if results[idx]["scores"][0]) if total else 0
    accuracy = round(correct / total, 6) if total else 0.0
    return accuracy


def score_prediction(doc: Dict, prediction: str) -> Dict[str, object]:
    res = mathvision_process_results(doc, [prediction])
    scores = res.get("mathvision_standard_eval", {}).get("scores", [])
    correct = bool(scores and scores[0])
    return {
        "correct": correct,
        "normalized_prediction": prediction.strip() if isinstance(prediction, str) else prediction,
        "normalized_target": doc.get("answer", ""),
    }


def aggregate_correct_list(correct_list: List[bool]) -> float:
    if not correct_list:
        return 0.0
    return round(sum(correct_list) / len(correct_list), 6)
