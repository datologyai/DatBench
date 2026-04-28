"""Tests for DatBench evaluator sample adaptation."""

import json
import unittest

from datbench import DatBenchEvaluator, VLMResponse


def _mcq_row(dataset: str) -> dict:
    metadata = {
        "options": ["alpha", "beta"],
        "index2ans": {"A": "alpha", "B": "beta"},
        "all_choices": ["A", "B"],
    }
    return {
        "id": f"{dataset}_sample",
        "image": None,
        "question": "<image 1>\nChoose the correct answer.",
        "prompt_format": {
            "prefix": "",
            "suffix": "\nA. alpha\nB. beta\nAnswer:",
        },
        "answer": "B",
        "all_answers": ["B"],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": dataset},
        "eval_metrics": {},
        "metadata": json.dumps(metadata),
    }


def _countbench_row() -> dict:
    return {
        "id": "countbench_sample",
        "image": None,
        "question": "<image 1>\nHow many objects are there?",
        "prompt_format": {
            "prefix": "",
            "suffix": "\nAnswer:",
        },
        "answer": "3",
        "all_answers": ["3"],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": "countbench"},
        "eval_metrics": {},
        "metadata": json.dumps({}),
    }


class DatBenchEvaluatorTest(unittest.TestCase):
    def test_mmmupro_receives_common_mcq_metadata(self) -> None:
        evaluator = DatBenchEvaluator([_mcq_row("mmmupro")], capability="general")

        report = evaluator.compute_metrics(
            [VLMResponse(id="mmmupro_sample", raw_output="\\boxed{B}")]
        )

        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        self.assertEqual(report.summary["dataset_metrics"]["mmmupro"]["accuracy"], 1.0)
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["pred_answer"], "B")

    def test_mmbench_still_receives_common_mcq_metadata(self) -> None:
        evaluator = DatBenchEvaluator([_mcq_row("mmbench")], capability="general")

        report = evaluator.compute_metrics(
            [VLMResponse(id="mmbench_sample", raw_output="\\boxed{B}")]
        )

        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        self.assertEqual(report.summary["dataset_metrics"]["mmbench"]["accuracy"], 1.0)
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["pred_answer"], "B")

    def test_common_mcq_metadata_is_optional_for_other_scorers(self) -> None:
        evaluator = DatBenchEvaluator([_countbench_row()], capability="counting")

        report = evaluator.compute_metrics(
            [VLMResponse(id="countbench_sample", raw_output="3")]
        )

        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["parsed_answer"], 3)


if __name__ == "__main__":
    unittest.main()
