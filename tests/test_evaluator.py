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


def _direct_row(
    dataset: str,
    question: str,
    answer: str,
    all_answers: list,
    metadata: dict,
    suffix: str = "\nAnswer:",
) -> dict:
    return {
        "id": f"{dataset}_sample",
        "image": None,
        "question": question,
        "prompt_format": {"prefix": "", "suffix": suffix},
        "answer": answer,
        "all_answers": all_answers,
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": dataset},
        "eval_metrics": {},
        "metadata": json.dumps(metadata),
    }


def _infovqa_row() -> dict:
    return _direct_row(
        "infovqa",
        "Which frame is highlighted?",
        "control frame",
        ["control frame", "control"],
        {},
    )


def _mathvista_multi_choice_row() -> dict:
    return _direct_row(
        "mathvista",
        "What is the value of x?\nA. 27\nB. 54\nC. 55\nD. 83\nAnswer:",
        "C",
        ["C"],
        {
            "question_type": "multi_choice",
            "choices": ["27", "54", "55", "83"],
            "all_choices": ["A", "B", "C", "D"],
        },
    )


def _mathvista_free_form_row() -> dict:
    return _direct_row(
        "mathvista",
        "How many cities have fewer than 70 billboards?\nAnswer:",
        "9",
        ["9"],
        {"question_type": "free_form", "choices": [], "all_choices": None},
    )


def _realworldqa_mcq_row() -> dict:
    return _direct_row(
        "realworldqa",
        "Is there a traffic light ahead?",
        "B",
        ["B"],
        {
            "answer_choices": [
                "Yes, green light ahead",
                "Yes, red light ahead",
                "Yes, yellow light ahead",
                "No, no traffic light",
            ]
        },
        suffix=(
            "\nA. Yes, green light ahead\nB. Yes, red light ahead"
            "\nC. Yes, yellow light ahead\nD. No, no traffic light\nAnswer:"
        ),
    )


def _realworldqa_free_form_row() -> dict:
    return _direct_row(
        "realworldqa",
        "How many motorcycles are in this image?",
        "2",
        ["2"],
        {"answer_choices": []},
    )


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

    def test_infovqa_scores_against_all_answers(self) -> None:
        evaluator = DatBenchEvaluator([_infovqa_row()], capability="chart")

        report = evaluator.compute_metrics(
            [VLMResponse(id="infovqa_sample", raw_output="CONTROL")]
        )

        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(
            score_details["all_ground_truth_answers"], ["control frame", "control"]
        )
        self.assertEqual(report.results[0].score, 1.0)
        self.assertEqual(
            report.summary["dataset_metrics"]["infovqa"]["accuracy__InfoVQA-ANLS"], 1.0
        )

    def test_mathvista_multi_choice_uses_letter_parsing(self) -> None:
        evaluator = DatBenchEvaluator([_mathvista_multi_choice_row()], capability="math")

        report = evaluator.compute_metrics(
            [
                VLMResponse(
                    id="mathvista_sample",
                    raw_output="Thus, the value of x is 55.\n\n**Answer:**\nC. 55",
                )
            ]
        )

        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["question_type"], "multi_choice")
        self.assertEqual(score_details["parsed_response"], "C")
        self.assertEqual(report.results[0].score, 1.0)

    def test_mathvista_free_form_null_all_choices_becomes_list(self) -> None:
        evaluator = DatBenchEvaluator([_mathvista_free_form_row()], capability="math")

        scorer_sample = evaluator._convert_sample_for_scorer(evaluator.samples[0])
        self.assertEqual(scorer_sample["question_type"], "free_form")
        self.assertEqual(scorer_sample["choices"], [])
        self.assertEqual(scorer_sample["all_choices"], [])

        report = evaluator.compute_metrics(
            [VLMResponse(id="mathvista_sample", raw_output="\\boxed{9}")]
        )

        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["question_type"], "free_form")
        self.assertEqual(report.results[0].score, 1.0)

    def test_realworldqa_maps_option_text_to_letter(self) -> None:
        evaluator = DatBenchEvaluator([_realworldqa_mcq_row()], capability="spatial")

        report = evaluator.compute_metrics(
            [VLMResponse(id="realworldqa_sample", raw_output="B. Yes, red light ahead")]
        )

        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["pred_answer"], "B")
        self.assertEqual(report.results[0].score, 1.0)

    def test_realworldqa_free_form_keeps_exact_match_path(self) -> None:
        evaluator = DatBenchEvaluator([_realworldqa_free_form_row()], capability="spatial")

        report = evaluator.compute_metrics(
            [VLMResponse(id="realworldqa_sample", raw_output="Two")]
        )

        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["pred_answer"], "2")
        self.assertEqual(report.results[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
