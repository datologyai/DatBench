"""Tests for ChartQA relaxed-correctness scoring."""

import json
import unittest

from datbench import DatBenchEvaluator, VLMResponse
from datbench.datasets_scoring import chartqa


def _score(prediction: str, gold: str) -> float:
    return chartqa.score_sample({"answer": gold}, prediction)["score"]


def _chartqa_row(row_id: str, answer: str) -> dict:
    return {
        "id": row_id,
        "image": None,
        "question": "<image 1>\nWhat is the value of the tallest bar?",
        "prompt_format": {
            "prefix": "",
            "suffix": "\nAnswer:",
        },
        "answer": answer,
        "all_answers": [answer],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": "chartqa"},
        "eval_metrics": {},
        "metadata": json.dumps({}),
    }


class ChartQARelaxedCorrectnessTest(unittest.TestCase):
    def test_wrong_numbers_are_not_rescued_by_text_match(self) -> None:
        # Sign and decimal-point errors used to pass once the text fallback
        # stripped "-" and ".".
        for prediction, gold in [
            ("-0.034", "0.034"),
            ("-33.8", "33.8"),
            ("1.6", "-1.6"),
            ("125", "1.25"),
            ("12.5%", "125"),
            ("-971", "971"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), 0.0)

    def test_numbers_within_tolerance_are_correct(self) -> None:
        for prediction, gold in [
            ("24.9", "24"),
            ("\\boxed{1.2}", "1.25"),
            ("0", "0"),
            ("0.0", "0"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), 1.0)

    def test_numbers_outside_tolerance_are_wrong(self) -> None:
        for prediction, gold in [
            ("25.3", "24"),
            ("1.32", "1.25"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), 0.0)

    def test_gold_zero_is_checked_numerically(self) -> None:
        self.assertEqual(_score("1", "0"), 0.0)
        self.assertEqual(_score("0.001", "0"), 0.0)

    def test_percent_and_thousands_conventions(self) -> None:
        for prediction, gold in [
            # Legacy reading: "X%" as X, and a ratio aligned to a percent.
            ("24%", "24"),
            ("0.24", "24%"),
            ("1,234", "1234"),
            # Official reading of a one-sided "%": "X%" as X / 100.
            ("302.38%", "3.0238"),
            ("250.5%", "2.5"),
            ("2.5", "250.5%"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), 1.0)

    def test_one_sided_percent_outside_both_readings_is_wrong(self) -> None:
        for prediction, gold in [
            ("30%", "3.0238"),
            ("12.5%", "125"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), 0.0)

    def test_text_and_list_answers_unchanged(self) -> None:
        for prediction, gold, expected in [
            ("United States.", "united states", 1.0),
            ("Yes", "No", 0.0),
            ("[Peru, Chile]", "[chile, peru]", 1.0),
            ("[Peru]", "[chile, peru]", 0.0),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_score(prediction, gold), expected)


class ChartQAEvaluatorTest(unittest.TestCase):
    def test_sign_error_scores_zero_through_evaluator(self) -> None:
        rows = [_chartqa_row("chartqa_sign", "971"), _chartqa_row("chartqa_ok", "24")]
        evaluator = DatBenchEvaluator(rows, capability="chart")

        report = evaluator.compute_metrics(
            [
                VLMResponse(id="chartqa_sign", raw_output="\\boxed{-971}"),
                VLMResponse(id="chartqa_ok", raw_output="\\boxed{24.5}"),
            ]
        )

        scores = {result.id: result.score for result in report.results}
        self.assertEqual(scores, {"chartqa_sign": 0.0, "chartqa_ok": 1.0})
        self.assertEqual(
            report.summary["dataset_metrics"]["chartqa"],
            {"accuracy__ChartQA-Relaxed": 0.5},
        )
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["pred_answer"], "-971")


if __name__ == "__main__":
    unittest.main()
