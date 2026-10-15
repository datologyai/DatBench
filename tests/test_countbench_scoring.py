"""Tests for CountBench answer extraction and scoring."""

import json
import unittest

from datbench import DatBenchEvaluator, VLMResponse
from datbench.datasets_scoring import countbench


def _countbench_row(answer: str) -> dict:
    return {
        "id": "countbench_sample",
        "image": None,
        "question": "<image 1>\nHow many dogs are there?",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": answer,
        "all_answers": [answer],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": "countbench"},
        "eval_metrics": {},
        "metadata": json.dumps({}),
    }


class CountBenchExtractNumberTest(unittest.TestCase):
    def test_bare_digits(self) -> None:
        self.assertEqual(countbench._extract_number("3"), 3)
        self.assertEqual(countbench._extract_number("12"), 12)

    def test_count_sentence_prefers_stated_count_over_trailing_digits(self) -> None:
        self.assertEqual(
            countbench._extract_number("There are three dogs and 2 cats."), 3
        )

    def test_final_answer_phrasing_wins_over_intermediate_numbers(self) -> None:
        text = "I see 2 on the left and 4 on the right. Therefore, the answer is 6."
        self.assertEqual(countbench._extract_number(text), 6)
        self.assertEqual(countbench._extract_number("Final answer: seven"), 7)

    def test_number_word_as_whole_token(self) -> None:
        self.assertEqual(countbench._extract_number("five"), 5)
        self.assertEqual(countbench._extract_number("Five."), 5)
        self.assertEqual(countbench._extract_number("eleven"), 11)

    def test_none_is_not_a_number(self) -> None:
        self.assertIsNone(countbench._extract_number("none"))
        self.assertIsNone(countbench._extract_number("I cannot tell."))
        self.assertIsNone(countbench._extract_number(""))

    def test_compound_number_words_are_not_resolved(self) -> None:
        self.assertIsNone(countbench._extract_number("twenty-one"))

    def test_runaway_digit_string_is_ignored(self) -> None:
        self.assertIsNone(countbench._extract_number("1" * 40))

    def test_last_digit_token_is_used(self) -> None:
        self.assertEqual(countbench._extract_number("maybe 4, actually 5"), 5)


class CountBenchScoreSampleTest(unittest.TestCase):
    def test_boxed_answer_scores_exact_match(self) -> None:
        details = countbench.score_sample({"answer": "3"}, "\\boxed{3}")
        self.assertEqual(details["score"], 1.0)
        self.assertEqual(details["parsed_answer"], 3)
        self.assertEqual(details["extracted_output"], "3")

    def test_count_sentence_scores_against_stated_count(self) -> None:
        details = countbench.score_sample(
            {"answer": "3"}, "There are three dogs and 2 cats."
        )
        self.assertEqual(details["score"], 1.0)
        self.assertEqual(details["parsed_answer"], 3)

    def test_no_number_scores_zero(self) -> None:
        details = countbench.score_sample({"answer": "3"}, "none")
        self.assertEqual(details["score"], 0.0)
        self.assertIsNone(details["parsed_answer"])

    def test_evaluator_end_to_end(self) -> None:
        evaluator = DatBenchEvaluator([_countbench_row("3")], capability="counting")

        report = evaluator.compute_metrics(
            [
                VLMResponse(
                    id="countbench_sample",
                    raw_output="There are three dogs and 2 cats.",
                )
            ]
        )

        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["parsed_answer"], 3)
        metrics = report.summary["dataset_metrics"]["countbench"]
        self.assertEqual(metrics["accuracy__CountBench-Accuracy"], 1.0)
        self.assertEqual(metrics["accuracy__CountBench-ParseRate"], 1.0)


if __name__ == "__main__":
    unittest.main()
