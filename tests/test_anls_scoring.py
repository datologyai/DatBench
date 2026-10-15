"""Tests for ANLS scoring in the DocVQA and InfoVQA scorers."""

import json
import sys
import types
import unittest
from unittest import mock

from datbench import DatBenchEvaluator, VLMResponse
from datbench.datasets_scoring import docvqa, infovqa


# (prediction, gold answers, expected ANLS)
ANLS_CASES = [
    # Normalized distance exactly 0.5: the official script keeps 0.5.
    ("16", ["1"], 0.5),
    ("2013", ["2007"], 0.5),
    ("5%", ["3%"], 0.5),
    # Distance > 0.5 is zeroed.
    ("1", ["234"], 0.0),
    ("2013", ["1987"], 0.0),
    # Near miss: 1 - 1/6.
    ("london", ["londom"], 5 / 6),
    ("Yes", ["yes"], 1.0),
    ("  New   York ", ["new york"], 1.0),
    # Unicode uppercase expansion is included in the length denominator.
    ("strabe", ["straße"], 6 / 7),
    ("straße", ["strabe"], 6 / 7),
    ("xxb", ["ﬃab"], 0.6),
    ("ﬃab", ["xxb"], 0.6),
    # Unicode expansion can also put normalized distance exactly at 0.5.
    ("xxb", ["ﬀab"], 0.5),
    ("ﬀab", ["xxb"], 0.5),
    # Max over gold answers.
    ("40%", ["60", "40 %", "40%"], 1.0),
    ("abcd", ["wxyz", "abcx"], 0.75),
]


def _infovqa_sample(answers: list) -> dict:
    return {"answers": answers}


def _docvqa_sample(answers: list) -> dict:
    return {"answer": answers[0], "all_answers": answers}


SCORERS = [
    ("infovqa", infovqa, _infovqa_sample),
    ("docvqa", docvqa, _docvqa_sample),
]


def _broken_anls_package() -> types.SimpleNamespace:
    """A stand-in for the PyPI `anls` package that fails if scoring calls it."""

    def anls_score(**kwargs):
        raise AssertionError("the anls package must not be used for scoring")

    return types.SimpleNamespace(anls_score=anls_score)


def _row(dataset: str, answer: str, all_answers: list) -> dict:
    return {
        "id": f"{dataset}_sample",
        "image": None,
        "question": "<image 1>\nWhat is written in the box?",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": answer,
        "all_answers": all_answers,
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": dataset},
        "eval_metrics": {},
        "metadata": json.dumps({}),
    }


class AnlsScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        # Scoring must not depend on the PyPI `anls` package (it zeroes a
        # similarity of exactly 0.5).
        patcher = mock.patch.dict(sys.modules, {"anls": _broken_anls_package()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_score_sample_matches_official_anls(self) -> None:
        for name, scorer, make_sample in SCORERS:
            for prediction, answers, expected in ANLS_CASES:
                with self.subTest(scorer=name, prediction=prediction, answers=answers):
                    result = scorer.score_sample(make_sample(answers), prediction)
                    self.assertAlmostEqual(result["score"], expected, places=12)

    def test_boxed_answer_is_extracted_before_scoring(self) -> None:
        for name, scorer, make_sample in SCORERS:
            with self.subTest(scorer=name):
                result = scorer.score_sample(
                    make_sample(["new york"]), "The city is \\boxed{New York}."
                )
                self.assertEqual(result["pred_answer"], "New York")
                self.assertEqual(result["score"], 1.0)

    def test_missing_gold_answers_score_zero(self) -> None:
        self.assertEqual(infovqa.score_sample({"answers": []}, "16")["score"], 0.0)
        self.assertEqual(
            docvqa.score_sample({"answer": "", "all_answers": []}, "16")["score"], 0.0
        )

    def test_infovqa_anls_key_matches_score(self) -> None:
        result = infovqa.score_sample(_infovqa_sample(["1"]), "16")
        self.assertEqual(result["anls"], 0.5)
        self.assertEqual(result["anls"], result["score"])


class DocVQAScoreDetailsTest(unittest.TestCase):
    def test_raw_similarity_is_kept_under_anls(self) -> None:
        result = docvqa.score_sample(
            _docvqa_sample(["Sodium is an essential nutrient."]), "Sodium"
        )

        self.assertEqual(result["score"], 0.0)
        self.assertGreater(result["anls"], 0.0)
        self.assertLess(result["anls"], 0.5)
        self.assertEqual(result["anls_thresholded_0_5"], 0.0)
        self.assertEqual(result["threshold_0_5_accuracy"], 0.0)

    def test_boundary_similarity_counts_as_correct(self) -> None:
        result = docvqa.score_sample(_docvqa_sample(["1"]), "16")

        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["anls"], 0.5)
        self.assertEqual(result["anls_thresholded_0_5"], 0.5)
        self.assertEqual(result["threshold_0_5_accuracy"], 1.0)

    def test_score_equals_raw_similarity_above_threshold(self) -> None:
        result = docvqa.score_sample(_docvqa_sample(["abcd"]), "abcx")

        self.assertAlmostEqual(result["score"], 0.75, places=12)
        self.assertEqual(result["score"], result["anls"])
        self.assertEqual(result["score"], result["anls_thresholded_0_5"])
        self.assertEqual(result["threshold_0_5_accuracy"], 1.0)

    def test_compute_metrics_reports_official_and_raw_means(self) -> None:
        details = [
            docvqa.score_sample(_docvqa_sample(["abcd"]), "axyz"),  # 0.25 raw -> 0
            docvqa.score_sample(_docvqa_sample(["abcd"]), "abcd"),  # exact match
        ]
        predictions = [{"score_details": sd} for sd in details]

        metrics = docvqa.compute_metrics(predictions)

        self.assertAlmostEqual(metrics["accuracy__DocVQA-ANLS"], 0.5, places=12)
        self.assertAlmostEqual(metrics["accuracy__DocVQA-ANLS-raw"], 0.625, places=12)
        self.assertAlmostEqual(
            metrics["accuracy__DocVQA-ANLS-thresholded-0.5"], 0.5, places=12
        )
        self.assertAlmostEqual(metrics["accuracy__DocVQA-threshold@0.5"], 0.5, places=12)
        self.assertAlmostEqual(metrics["accuracy__DocVQA-ExactMatch"], 0.5, places=12)


class InfoVQAMetricsTest(unittest.TestCase):
    def test_compute_metrics_averages_anls_and_exact_match(self) -> None:
        details = [
            infovqa.score_sample(_infovqa_sample(["1"]), "16"),  # boundary 0.5
            infovqa.score_sample(_infovqa_sample(["yes"]), "Yes"),  # exact match
        ]
        predictions = [{"score_details": sd} for sd in details]

        metrics = infovqa.compute_metrics(predictions)

        self.assertAlmostEqual(metrics["accuracy__InfoVQA-ANLS"], 0.75, places=12)
        self.assertAlmostEqual(metrics["accuracy__InfoVQA-ExactMatch"], 0.5, places=12)


class EvaluatorAnlsTest(unittest.TestCase):
    def test_docvqa_sample_score_is_official_anls(self) -> None:
        row = _row("docvqa", "Sodium is an essential nutrient.", ["Sodium is an essential nutrient."])
        evaluator = DatBenchEvaluator([row], capability="document")

        report = evaluator.compute_metrics(
            [VLMResponse(id="docvqa_sample", raw_output="\\boxed{Sodium}")]
        )

        result = report.results[0]
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.is_correct)
        self.assertGreater(result.metadata["score_details"]["anls"], 0.0)
        dataset_metrics = report.summary["dataset_metrics"]["docvqa"]
        self.assertEqual(dataset_metrics["accuracy__DocVQA-ANLS"], 0.0)
        self.assertGreater(dataset_metrics["accuracy__DocVQA-ANLS-raw"], 0.0)

    def test_infovqa_boundary_sample_is_correct(self) -> None:
        row = _row("infovqa", "1", ["1"])
        evaluator = DatBenchEvaluator([row], capability="chart")

        report = evaluator.compute_metrics(
            [VLMResponse(id="infovqa_sample", raw_output="\\boxed{16}")]
        )

        result = report.results[0]
        self.assertEqual(result.score, 0.5)
        self.assertTrue(result.is_correct)
        self.assertEqual(
            report.summary["dataset_metrics"]["infovqa"]["accuracy__InfoVQA-ANLS"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
