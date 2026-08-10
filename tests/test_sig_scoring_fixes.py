"""Format-robustness fixes for extraction/grounding/spotting scorers.

Four defects surfaced by rescoring the DAT-3459 arms from saved predictions:

1. Text spotting was "simplified to VQA": the box list was string-matched
   against the flattened polygon answer, which can essentially never match,
   so every run scored a hard 0. Replaced with the official RRC-style
   single-image metric (greedy IoU > 0.5 match + recognition h-mean).
2. KIE F1 zeroed predictions that keep the GT's own single-element list
   syntax ({'key': ['value']}) — the ground truth fed back as a prediction
   scored 0.0.
3. VQA-with-position/grounding rows in exported eval sets carry the GT box
   only as four numeric answer strings (bbox field is null), so the IoU half
   of the 0.5*content + 0.5*IoU score could never be earned.
4. CC-OCR KIE returned an empty dict when a model merged the \\boxed braces
   with the JSON braces (\\boxed{"k": "v"}).
"""

from datbench.datasets_scoring import cc_ocr_kie
from datbench.datasets_scoring import ocrbench_v2 as ob

SPOTTING_GT = {
    "bbox_list": [
        [10, 10, 110, 10, 110, 60, 10, 60],
        [200, 200, 300, 200, 300, 260, 200, 260],
    ],
    "content": ["HELLO", "WORLD"],
}


def test_spotting_perfect_prediction_scores_one():
    pred = "[(10, 10, 110, 60, 'HELLO'), (200, 200, 300, 260, 'WORLD')]"
    assert ob._text_spotting_evaluation(pred, SPOTTING_GT) == 1.0


def test_spotting_boxed_prediction_scores_one():
    pred = "\\boxed{[(10, 10, 110, 60, 'HELLO'), (200, 200, 300, 260, 'WORLD')]}"
    assert ob._text_spotting_evaluation(pred, SPOTTING_GT) == 1.0


def test_spotting_fabricated_boxes_score_zero():
    pred = "[(500, 500, 600, 600, 'HELLO'), (700, 700, 800, 800, 'WORLD')]"
    assert ob._text_spotting_evaluation(pred, SPOTTING_GT) == 0.0


def test_spotting_wrong_text_scores_zero():
    pred = "[(10, 10, 110, 60, 'GOODBYE'), (200, 200, 300, 260, 'MOON')]"
    assert ob._text_spotting_evaluation(pred, SPOTTING_GT) == 0.0


def test_spotting_ignores_hashhash_regions():
    gt = {"bbox_list": SPOTTING_GT["bbox_list"], "content": ["HELLO", "###"]}
    pred = "[(10, 10, 110, 60, 'HELLO')]"
    assert ob._text_spotting_evaluation(pred, gt) == 1.0


GT_LIST_SYNTAX = "{'CE-P1': ['1700kJ/410kcal'], 'TF-P1': ['35g'], 'PRO-P1': ['24g']}"


def test_kie_f1_gt_fed_back_as_prediction_scores_one():
    score = ob._kie_f1_evaluation(
        GT_LIST_SYNTAX, [GT_LIST_SYNTAX], "key information extraction en"
    )
    assert score == 1.0


def test_kie_f1_scalar_syntax_still_scores_one():
    scalar = "{'CE-P1': '1700kJ/410kcal', 'TF-P1': '35g', 'PRO-P1': '24g'}"
    score = ob._kie_f1_evaluation(
        scalar, [GT_LIST_SYNTAX], "key information extraction en"
    )
    assert score == 1.0


def test_kie_f1_wrong_value_still_scores_zero():
    score = ob._kie_f1_evaluation(
        "{'CE-P1': ['9999']}",
        ["{'CE-P1': ['1700kJ/410kcal']}"],
        "key information extraction en",
    )
    assert score == 0.0


def test_bbox_from_answers_rebuilds_box():
    assert ob._bbox_from_answers(["630", "443", "828", "496"]) == [630, 443, 828, 496]
    assert ob._bbox_from_answers(["Walter Crickmer"]) is None
    assert ob._bbox_from_answers(["a", "b", "c", "d"]) is None


def test_vqa_with_position_scores_iou_when_bbox_field_is_null():
    sample = {"bbox": None, "answers": ["630", "443", "828", "496"]}
    score = ob._vqa_with_position_evaluation("(630, 443, 828, 496)", sample)
    assert score >= 0.5


def test_vqa_with_position_prefers_explicit_bbox_field():
    pred = "(630, 443, 828, 496)"
    answers = ["630", "443", "828", "496"]
    hit = ob._vqa_with_position_evaluation(
        pred, {"bbox": [630, 443, 828, 496], "answers": answers}
    )
    miss = ob._vqa_with_position_evaluation(
        pred, {"bbox": [0, 0, 10, 10], "answers": answers}
    )
    # if the answers fallback overrode the explicit bbox, both samples would
    # score identically; the explicit box must decide the IoU half instead
    assert hit - miss == 0.5


def test_cc_ocr_extract_json_boxed_merged_braces():
    assert cc_ocr_kie._extract_json('\\boxed{"CE-PS": "110", "TF-PS": "1g"}') == {
        "CE-PS": "110",
        "TF-PS": "1g",
    }


def test_cc_ocr_extract_json_boxed_with_proper_braces_unchanged():
    assert cc_ocr_kie._extract_json('\\boxed{{"CE-PS": "110"}}') == {"CE-PS": "110"}


def test_cc_ocr_extract_json_bare_json_unchanged():
    assert cc_ocr_kie._extract_json('{"CE-PS": "110"}') == {"CE-PS": "110"}
