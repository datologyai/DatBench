"""OCRBench v2 scoring.

Multi-task OCR benchmark with 40+ task types across 13 high-level categories.
"""

import ast
import json
import re
from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single OCRBench v2 sample.

    Routes to appropriate scoring function based on task type.

    Args:
        sample: Sample data with task type information
        model_output: Model's generated output

    Returns:
        Dictionary containing score and task metadata
    """
    task_type = sample.get("type", "")
    answers = sample.get("answers") or sample.get("all_answers", [])
    eval_method = sample.get("eval")

    # For most tasks, extract short answer
    # BUT for structured output tasks (KIE, parsing), use raw output
    structured_output_tasks = [
        "key information extraction en", "key information mapping en",
        "key information extraction cn", "chart parsing en",
        "document parsing en", "document parsing cn",
        "table parsing en", "table parsing cn",
        "text grounding en", "VQA with position en",
        "text spotting en",
    ]

    if task_type in structured_output_tasks:
        pred_answer = model_output
    else:
        pred_answer = extract_final_answer(model_output)

    # Route to appropriate scoring function based on task type
    score = 0.0
    scoring_method = "unknown"

    #===============================================================
    # GROUP 1: English VQA Tasks (most common)
    #===============================================================
    if task_type in [
        "APP agent en", "ASCII art classification en", "reasoning VQA en",
        "science QA en", "text recognition en", "document classification en",
        "cognition VQA en", "diagram QA en", "fine-grained text recognition en"
    ]:
        if eval_method == "multiple choice":
            # Extract only alphabetic characters
            if not isinstance(pred_answer, str):
                score = 0.0
            else:
                predict_letters = ''.join(c for c in pred_answer if c.isalpha())
                if len(answers) == 1 and predict_letters == answers[0]:
                    score = 1.0
                else:
                    score = 0.0
            scoring_method = "multiple_choice"
        elif eval_method == "case sensitive":
            score = _vqa_evaluation_case_sensitive(pred_answer, answers)
            scoring_method = "vqa_case_sensitive"
        else:
            score = _vqa_evaluation(pred_answer, answers)
            scoring_method = "vqa"

    #===============================================================
    # GROUP 2: Chinese VQA Tasks
    #===============================================================
    elif task_type in ["cognition VQA cn", "reasoning VQA cn"]:
        if eval_method == "multiple choice":
            if not isinstance(pred_answer, str):
                score = 0.0
            else:
                predict_letters = ''.join(c for c in pred_answer if c.isalpha())
                if len(answers) == 1 and predict_letters == answers[0]:
                    score = 1.0
                else:
                    score = 0.0
            scoring_method = "multiple_choice"
        elif eval_method == "case sensitive":
            score = _vqa_evaluation_case_sensitive(pred_answer, answers)
            scoring_method = "vqa_case_sensitive"
        else:
            score = _cn_vqa_evaluation(pred_answer, answers)
            scoring_method = "cn_vqa"

    #===============================================================
    # GROUP 3: Math Expression Tasks
    #===============================================================
    elif task_type in ["formula recognition en"]:
        score = _math_expression_evaluation(pred_answer, answers)
        scoring_method = "math_expression"

    elif task_type in ["formula recognition cn"]:
        score = _cn_math_expression_evaluation(pred_answer, answers)
        scoring_method = "cn_math_expression"

    #===============================================================
    # GROUP 4: Math QA
    #===============================================================
    elif task_type == "math QA en":
        if eval_method == "multiple choice":
            if not isinstance(pred_answer, str):
                score = 0.0
            else:
                predict_letters = ''.join(c for c in pred_answer if c.isalpha())
                if len(answers) == 1 and predict_letters == answers[0]:
                    score = 1.0
                else:
                    score = 0.0
            scoring_method = "multiple_choice"
        else:
            score = _vqa_evaluation(pred_answer, answers)
            scoring_method = "vqa"

    #===============================================================
    # GROUP 5: Counting Tasks
    #===============================================================
    elif task_type == "text counting en":
        count_eval_method = eval_method if eval_method else "exact match"
        score = _counting_evaluation(pred_answer, answers, count_eval_method)
        scoring_method = "counting"

    #===============================================================
    # GROUP 6: KIE (Key Information Extraction)
    #===============================================================
    elif task_type in ["key information extraction en", "key information mapping en", "key information extraction cn"]:
        score = _kie_f1_evaluation(pred_answer, answers, task_type)
        scoring_method = "kie_f1"

    #===============================================================
    # GROUP 7: Table Parsing (TEDS)
    #===============================================================
    elif task_type in ["table parsing en", "table parsing cn"]:
        score = _table_parsing_evaluation(pred_answer, answers, sample.get("question", ""))
        scoring_method = "teds"

    #===============================================================
    # GROUP 8: Document/Chart Parsing
    #===============================================================
    elif task_type in ["document parsing en", "document parsing cn"]:
        score = _doc_parsing_evaluation(pred_answer, answers)
        scoring_method = "doc_parsing_f1"

    elif task_type == "chart parsing en":
        score = _chart_parsing_evaluation(pred_answer, answers)
        scoring_method = "chart_parsing_f1"

    #===============================================================
    # GROUP 9: Position/Grounding Tasks (IoU)
    #===============================================================
    elif task_type in ["text grounding en", "VQA with position en"]:
        score = _vqa_with_position_evaluation(pred_answer, sample)
        scoring_method = "vqa_with_position"

    #===============================================================
    # GROUP 10: Text Spotting (bbox + text)
    #===============================================================
    elif task_type == "text spotting en":
        score = _text_spotting_evaluation(pred_answer, sample)
        scoring_method = "text_spotting"

    #===============================================================
    # GROUP 11: Full-Page OCR
    #===============================================================
    elif task_type in ["full-page OCR en", "full-page OCR cn"]:
        score = _full_page_ocr_evaluation(pred_answer, answers)
        scoring_method = "full_page_ocr"

    #===============================================================
    # GROUP 12: Handwritten Answer Extraction
    #===============================================================
    elif task_type == "handwritten answer extraction cn":
        score = _handwritten_extraction_evaluation(pred_answer, answers, sample.get("question", ""))
        scoring_method = "handwritten_extraction"

    #===============================================================
    # GROUP 13: Text Translation
    #===============================================================
    elif task_type == "text translation cn":
        score = _cn_vqa_evaluation(pred_answer, answers)
        scoring_method = "cn_vqa"

    else:
        # Unknown task type - use vqa_evaluation as fallback
        score = _vqa_evaluation(pred_answer, answers)
        scoring_method = "vqa_fallback"

    return {
        "score": score,
        "task_type": task_type,
        "dataset_name": sample.get("dataset_name", ""),
        "scoring_method": scoring_method,
        "ground_truth": answers,
        "model_output": model_output,
        "pred_answer": pred_answer,
    }


#===========================================================================
# CORE VQA SCORING FUNCTIONS (from official vqa_metric.py)
#===========================================================================

def _vqa_evaluation(predict: str, answers: List[str]) -> float:
    """VQA evaluation for OCRBench v2."""
    score = 0
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, (int, float)):
                answer = str(answer)
            try:
                answer = answer.lower().strip().replace("\n", " ")
            except:
                continue

            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict_norm = predict.lower().strip().replace("\n", " ")

            if len(answer.split()) < 5:
                # Short answer: substring match
                if answer in predict_norm:
                    score = 1
            else:
                # Long answer: ANLS
                dist = _levenshtein_distance(predict_norm, answer)
                length = max(len(predict_norm), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value
    else:
        # Single answer (not list)
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
        else:
            dist = _levenshtein_distance(predict, answers)
            length = max(len(predict), len(answers))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def _cn_vqa_evaluation(predict: str, answers: List[str]) -> float:
    """Chinese VQA evaluation (removes all spaces).
    """
    score = 0
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, (int, float)):
                answer = str(answer)
            try:
                answer = answer.lower().strip().replace("\n", " ").replace(" ", "")
            except:
                continue

            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict_norm = predict.lower().strip().replace("\n", " ").replace(" ", "")

            if len(answer.split(",")) < 4:
                # Short answer: substring match
                if answer in predict_norm:
                    score = 1
            else:
                # Long answer: ANLS
                dist = _levenshtein_distance(predict_norm, answer)
                length = max(len(predict_norm), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value
    else:
        answers = answers.lower().strip().replace("\n", " ").replace(" ", "")
        predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
        if len(answers.split(",")) < 4:
            if answers in predict:
                score = 1
        else:
            dist = _levenshtein_distance(predict, answers)
            length = max(len(predict), len(answers))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def _vqa_evaluation_case_sensitive(predict: str, answers: List[str]) -> float:
    """Case-sensitive VQA evaluation."""
    score = 0
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, (int, float)):
                answer = str(answer)
            try:
                answer = answer.strip().replace("\n", " ")
            except:
                continue
            predict_norm = predict.strip().replace("\n", " ")

            if len(answer.split()) < 5:
                if answer in predict_norm:
                    score = 1
            else:
                dist = _levenshtein_distance(predict_norm, answer)
                length = max(len(predict_norm), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value
    else:
        answers = answers.strip().replace("\n", " ")
        predict = predict.strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
        else:
            dist = _levenshtein_distance(predict, answers)
            length = max(len(predict), len(answers))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def _math_expression_evaluation(predict: str, answers: List[str]) -> float:
    """Math expression evaluation (substring match, no spaces).
    """
    score = 0
    if isinstance(answers, list):
        for answer in answers:
            answer_norm = answer.strip().replace("\n", " ").replace(" ", "")
            predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
            if answer_norm in predict_norm:
                score = 1
    else:
        answers_norm = answers.strip().replace("\n", " ").replace(" ", "")
        predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
        if answers_norm in predict_norm:
            score = 1
    return score


def _cn_math_expression_evaluation(predict: str, answers: List[str]) -> float:
    """Chinese math expression evaluation (removes \\text{} tags).
    """
    score = 0

    # Remove LaTeX text tags
    def remove_text_tags(latex_str):
        pattern = r'\\text\{([^{}]*)\}'
        return re.sub(pattern, r'\1', latex_str)

    if len(answers) == 1:
        answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)

    if isinstance(answers, list):
        for answer in answers:
            answer_norm = answer.strip().replace("\n", " ").replace(" ", "")
            predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
            if answer_norm in predict_norm:
                score = 1
    else:
        answers_norm = answers.strip().replace("\n", " ").replace(" ", "")
        predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
        if answers_norm in predict_norm:
            score = 1

    return score


def _counting_evaluation(predict: str, answers: List[str], eval_method: str) -> float:
    """Counting evaluation (IoU on extracted numbers).
    """
    score = 0
    temp_score = 0

    # Extract first number from prediction
    def extract_first_number(text):
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else None

    if isinstance(answers, list):
        for answer in answers:
            if eval_method == "exact match":
                answer_norm = answer.lower().strip().replace("\n", " ")
                predict_norm = predict.lower().strip().replace("\n", " ")
                if answer_norm in predict_norm:
                    score = 1
            elif eval_method == "regression":
                predict_number = extract_first_number(predict)
                if predict_number is not None:
                    try:
                        answer_num = int(answer)
                        if predict_number <= 0 or predict_number >= 2 * answer_num:
                            score = 0
                        else:
                            iou = 1 - abs(predict_number - answer_num) / answer_num
                            score = iou if iou > 0.5 else 0
                    except:
                        score = 0
                else:
                    score = 0
            else:
                answer_norm = answer.lower().strip().replace("\n", " ")
                predict_norm = predict.lower().strip().replace("\n", " ")
                if answer_norm in predict_norm:
                    score = 1

            if score > temp_score:
                temp_score = score
        score = temp_score
    else:
        # Single answer
        if eval_method == "exact match":
            if answers.lower().strip() in predict.lower().strip():
                score = 1
        elif eval_method == "regression":
            predict_number = extract_first_number(predict)
            if predict_number is not None:
                try:
                    answer_num = int(answers)
                    if predict_number > 0 and predict_number < 2 * answer_num:
                        iou = 1 - abs(predict_number - answer_num) / answer_num
                        score = iou if iou > 0.5 else 0
                except:
                    score = 0

    return score


def _kie_f1_evaluation(predict: str, answers: List[str], task_type: str) -> float:
    """KIE F1 evaluation with dict extraction and field-level F1."""
    if not answers:
        return 0.0

    # Extract dict from prediction
    pred_dict = _extract_dict_from_text(predict)
    if not isinstance(pred_dict, dict):
        return 0.0

    # Parse ground truth
    try:
        if isinstance(answers[0], str):
            gt_raw = ast.literal_eval(answers[0])
        else:
            gt_raw = answers[0]

        if not isinstance(gt_raw, dict):
            return 0.0

        # Generate all combinations (GT values can be lists)
        gts = _generate_combinations(gt_raw)

        if not gts:
            return 0.0

    except:
        return 0.0

    # Take max score across all GT variations
    max_score = 0.0
    for gt_dict in gts:
        if isinstance(gt_dict, dict):
            score = _compute_kie_f1(pred_dict, gt_dict)
            max_score = max(max_score, score)

    return max_score


def _extract_dict_from_text(text: str) -> Dict[str, Any]:
    """Extract dictionary from text."""
    # Remove code fences
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = text.strip()

    # Try JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except:
        pass

    # Try ast.literal_eval
    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            return data
    except:
        pass

    # Try regex parsing
    data = {}
    key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
    matches = re.findall(key_value_pattern, content)
    for key, value in matches:
        data[key.strip()] = value.strip()

    return data if isinstance(data, dict) else {}


def _generate_combinations(input_dict: Dict) -> List[Dict]:
    """Generate all combinations from dict with list values."""
    from itertools import product

    # Ensure all values are lists
    processed_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, list):
            processed_dict[k] = v
        else:
            processed_dict[k] = [v]

    # Generate all combinations
    keys = list(processed_dict.keys())
    value_lists = [processed_dict[k] for k in keys]

    if not keys:
        return [{}]

    combinations = []
    for values_tuple in product(*value_lists):
        combo = {keys[i]: values_tuple[i] for i in range(len(keys))}
        combinations.append(combo)

    return combinations if combinations else [{}]


def _compute_kie_f1(pred_dict: Dict, gt_dict: Dict) -> float:
    """Compute field-level F1 score for KIE."""
    keys = set(pred_dict.keys()).union(set(gt_dict.keys()))

    tp = 0
    fp = 0
    fn = 0

    for key in keys:
        pred_val = pred_dict.get(key)
        gt_val = gt_dict.get(key)

        # Normalize values
        if pred_val:
            pred_val = str(pred_val).lower().strip().replace("\n", " ").replace(" ", "")
        if gt_val:
            gt_val = str(gt_val).lower().strip().replace("\n", " ").replace(" ", "")

        if pred_val is None and gt_val is None:
            continue
        elif pred_val is None:
            fn += 1
        elif gt_val is None:
            fp += 1
        else:
            if pred_val == gt_val:
                tp += 1
            else:
                fp += 1
                fn += 1

    # Compute F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def _table_parsing_evaluation(predict: str, answers: List[str], question: str) -> float:
    """Table parsing with TEDS evaluation."""
    if not answers or not isinstance(predict, str):
        return 0.0

    try:
        # Check if question asks for HTML
        if "html" in question.lower():
            # Extract HTML table from prediction
            predict_table = predict.replace('\n', '')

            # Find table content
            if "<body" in predict_table:
                predict_table = re.findall('<body.*', predict_table)[0]
            elif "<table" in predict_table:
                predict_table = re.findall('<table.*', predict_table)[0]
            else:
                return 0.0

            # Wrap with required tags
            predict_table = _wrap_html_table(predict_table)
            gt_table = _wrap_html_table(answers[0])

            # Use official OCRBench v2 TEDS
            try:
                from .evaluation_utils.ocrbench_v2_teds import TEDS

                teds = TEDS(structure_only=False, n_jobs=1)
                score = teds.evaluate(predict_table, gt_table)

                return max(0.0, min(1.0, score))

            except ImportError:
                return 0.0
        else:
            # Markdown table - fall back to VQA
            return _vqa_evaluation(predict, answers)

    except Exception:
        return 0.0


def _wrap_html_table(html_table: str) -> str:
    """Wrap HTML table with required tags."""
    html_table = html_table.replace('\n', '')

    # Add missing <table> tags
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"

    # Add <body> tags
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'

    # Add <html> tags
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'

    return html_table


def _doc_parsing_evaluation(predict: str, answers: List[str]) -> float:
    """Document parsing - simplified (use KIE-like F1)."""
    return _kie_f1_evaluation(predict, answers, "doc_parsing")


def _chart_parsing_evaluation(predict: str, answers: List[str]) -> float:
    """Chart parsing - simplified (use KIE-like F1)."""
    return _kie_f1_evaluation(predict, answers, "chart_parsing")


def _vqa_with_position_evaluation(predict: str, sample: Dict[str, Any]) -> float:
    """VQA with position: 50% answer + 50% bbox IoU.
    """
    import ast

    score_content = 0.0
    score_bbox = 0.0

    # Try to parse as structured dict with "answer" and "bbox"
    pred_dict = _try_parse_as_dict(predict)

    if isinstance(pred_dict, dict) and "answer" in pred_dict:
        # Structured output
        score_content = _vqa_evaluation(pred_dict["answer"], sample.get("answers", []))

        if "bbox" in pred_dict and sample.get("bbox"):
            try:
                pred_bbox = ast.literal_eval(str(pred_dict["bbox"]))
                score_bbox = _calculate_iou(pred_bbox, sample["bbox"])
            except:
                score_bbox = 0.0
    else:
        # Free-form text - extract coordinates
        coords = _extract_coordinates(predict)
        if coords and sample.get("bbox"):
            score_bbox = _calculate_iou(coords, sample["bbox"])

        # Score the answer content
        score_content = _vqa_evaluation(predict, sample.get("answers", []))

    # Weighted combination: 50% content, 50% bbox
    return 0.5 * score_content + 0.5 * score_bbox


def _text_spotting_evaluation(predict: str, sample: Dict[str, Any]) -> float:
    """Text spotting - simplified to VQA."""
    answers = sample.get("answers", [])
    return _vqa_evaluation(predict, answers)


def _full_page_ocr_evaluation(predict: str, answers: List[str]) -> float:
    """Full-page OCR: BLEU + METEOR + F1 + (1 - edit_dist) / 4."""
    if not answers:
        return 0.0

    gt = answers[0] if isinstance(answers, list) else answers
    pred = predict.strip()
    gt = gt.strip()

    # Detect Chinese
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    has_chinese = bool(chinese_pattern.search(gt)) or bool(chinese_pattern.search(pred))

    try:
        if has_chinese:
            try:
                import jieba
                reference = jieba.lcut(gt)
                hypothesis = jieba.lcut(pred)
            except:
                reference = list(gt)
                hypothesis = list(pred)
        else:
            reference = gt.split()
            hypothesis = pred.split()

        # BLEU
        try:
            import nltk
            bleu = nltk.translate.bleu([reference], hypothesis)
        except:
            bleu = 0.0

        # METEOR
        try:
            from nltk.translate import meteor_score
            meteor = meteor_score.meteor_score([reference], hypothesis)
        except:
            meteor = 0.0

        # F-measure on word sets
        try:
            from nltk.metrics import f_measure
            ref_set = set(reference)
            hyp_set = set(hypothesis)
            f_score = f_measure(ref_set, hyp_set) if (ref_set or hyp_set) else 0.0
            if f_score is None:
                f_score = 0.0
        except:
            f_score = 0.0

        # Edit distance
        edit_dist = _levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        edit_dist_norm = edit_dist / max_len if max_len > 0 else 0.0

        # Average: (BLEU + METEOR + F1 + (1 - edit_dist)) / 4
        score = (bleu + meteor + f_score + (1 - edit_dist_norm)) / 4.0

        return max(0.0, min(1.0, score))

    except Exception:
        # Fallback to ANLS
        dist = _levenshtein_distance(pred, gt)
        length = max(len(pred), len(gt))
        if length == 0:
            return 1.0
        ANLS = 1 - (dist / length)
        return ANLS if ANLS >= 0.5 else 0.0


def _handwritten_extraction_evaluation(predict: str, answers: List[str], question: str) -> float:
    """Handwritten answer extraction."""
    # Check if it's a short answer question
    if "简答" in question:
        return _full_page_ocr_evaluation(predict, answers)
    else:
        return _cn_vqa_evaluation(predict, answers)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance.

    
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute OCRBench v2 metrics with category-based aggregation.

    Aggregates scores by high-level category, then averages categories
    to match official leaderboard methodology from get_score.py
    """
    if not results:
        return {"accuracy": 0.0}

    # Initialize category score lists (from official get_score.py)
    en_text_recognition_list = []
    en_text_detection_list = []
    en_text_spotting_list = []
    en_relationship_extraction_list = []
    en_element_parsing_list = []
    en_mathematical_calculation_list = []
    en_visual_text_understanding_list = []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list = []
    cn_relationship_extraction_list = []
    cn_element_parsing_list = []
    cn_visual_text_understanding_list = []
    cn_knowledge_reasoning_list = []

    # Group scores by category (exact logic from get_score.py)
    for result in results:
        score_details = result.get("score_details", result)
        task_type = score_details.get("task_type", "")
        score = score_details.get("score", 0.0)

        # English categories
        if task_type in ["text recognition en", "fine-grained text recognition en", "full-page OCR en"]:
            en_text_recognition_list.append(score)

        elif task_type in ["text grounding en", "VQA with position en"]:
            en_text_detection_list.append(score)

        elif task_type == "text spotting en":
            en_text_spotting_list.append(score)

        elif task_type in ["key information extraction en", "key information mapping en"]:
            en_relationship_extraction_list.append(score)

        elif task_type in ["document parsing en", "chart parsing en", "table parsing en", "formula recognition en"]:
            en_element_parsing_list.append(score)

        elif task_type in ["math QA en", "text counting en"]:
            en_mathematical_calculation_list.append(score)

        elif task_type in ["document classification en", "cognition VQA en", "diagram QA en"]:
            en_visual_text_understanding_list.append(score)

        elif task_type in ["reasoning VQA en", "science QA en", "APP agent en", "ASCII art classification en"]:
            en_knowledge_reasoning_list.append(score)

        # Chinese categories
        elif task_type == "full-page OCR cn":
            cn_text_recognition_list.append(score)

        elif task_type in ["key information extraction cn", "handwritten answer extraction cn"]:
            cn_relationship_extraction_list.append(score)

        elif task_type in ["document parsing cn", "table parsing cn", "formula recognition cn"]:
            en_element_parsing_list.append(score)

        elif task_type == "cognition VQA cn":
            cn_visual_text_understanding_list.append(score)

        elif task_type in ["reasoning VQA cn", "text translation cn"]:
            cn_knowledge_reasoning_list.append(score)

    # Compute category averages
    def safe_mean(scores_list):
        return sum(scores_list) / len(scores_list) if scores_list else 0.0

    en_scores = {
        "Recognition": safe_mean(en_text_recognition_list),
        "Referring": safe_mean(en_text_detection_list),
        "Spotting": safe_mean(en_text_spotting_list),
        "Extraction": safe_mean(en_relationship_extraction_list),
        "Parsing": safe_mean(en_element_parsing_list),
        "Calculation": safe_mean(en_mathematical_calculation_list),
        "Understanding": safe_mean(en_visual_text_understanding_list),
        "Reasoning": safe_mean(en_knowledge_reasoning_list),
    }

    cn_scores = {
        "CN_Recognition": safe_mean(cn_text_recognition_list),
        "CN_Extraction": safe_mean(cn_relationship_extraction_list),
        "CN_Parsing": safe_mean(cn_element_parsing_list),
        "CN_Understanding": safe_mean(cn_visual_text_understanding_list),
        "CN_Reasoning": safe_mean(cn_knowledge_reasoning_list),
    }

    # Overall averages
    en_overall = sum(en_scores.values()) / len(en_scores) if en_scores else 0.0
    cn_overall = sum(cn_scores.values()) / len(cn_scores) if cn_scores else 0.0

    # Return metrics matching official leaderboard format
    return {
        **en_scores,
        **cn_scores,
        "Average": en_overall,
        "CN_Average": cn_overall,
        "accuracy": en_overall,
        "score": en_overall,
    }


# Helper functions for VQA with position

def _try_parse_as_dict(text: str):
    """Try to parse text as dictionary.


    """
    import ast

    try:
        return json.loads(text)
    except:
        pass

    try:
        return ast.literal_eval(text)
    except:
        pass

    return None


def _extract_coordinates(text: str):
    """Extract coordinates (x1, y1, x2, y2) from text.


    """
    # Pattern: (x1, y1, x2, y2) or [x1, y1, x2, y2]
    pattern = r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]'

    matches = list(re.finditer(pattern, text))
    coords_list = []
    coords_set = set()

    for match in matches:
        x1, y1, x2, y2 = map(int, match.groups())

        # Validate coordinates (0-1000 range)
        if all(0 <= n <= 1000 for n in [x1, y1, x2, y2]):
            coords = (x1, y1, x2, y2)
            if coords not in coords_set:
                coords_list.append(coords)
                coords_set.add(coords)

    # Return last valid coordinates
    if coords_list:
        return list(coords_list[-1])
    return None


def _calculate_iou(box1, box2) -> float:
    """Calculate IoU between two bounding boxes.


    """
    try:
        box1 = [int(c) for c in box1]
        box2 = [int(c) for c in box2]
    except:
        return 0.0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0.0

    return iou
