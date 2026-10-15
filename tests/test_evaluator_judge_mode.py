"""Tests for judge-mode scoring in the DatBench evaluator."""

import json
import unittest
import warnings

from datbench import (
    DatBenchEvaluator,
    JudgeRequest,
    JudgeResponse,
    VLMResponse,
    parse_judge_verdict,
)
from datbench.schema import DatBenchSample


TWO_LINE_JUDGE_PROMPT = (
    "Rubric: the response is correct if it names the same value as the reference.\n\n"
    "Respond with exactly two lines:\n"
    "Line 1: Reason: <short justification>\n"
    "Line 2: answer: true or false"
)
SINGLE_TOKEN_SUFFIX = "\n\nOutput exactly one boolean token: true or false. answer: "

JUDGE_SCORE_DETAIL_KEYS = {
    "score",
    "eval_mode",
    "match",
    "judge_verdict",
    "judge_raw_output",
    "judge_missing_response",
}
JUDGE_DATASET_METRIC_KEYS = {
    "judge_accuracy",
    "judge_samples",
    "judge_missing_responses",
    "judge_unparsed_verdicts",
}
SAMPLE_SCORE_METADATA_KEYS = {
    "capability",
    "eval_mode",
    "dataset",
    "is_circular",
    "discrimination",
    "score_details",
}

RAW_OUTPUT_CASES = [
    ('{"answer": true}', True),
    ('{"answer": false}', False),
    ('{"answer": "Yes"}', True),
    ('{"answer": "incorrect"}', False),
    ('```json\n{"answer": true}\n```', True),
    ("answer: true", True),
    ("Answer: False", False),
    ("Reason: values match.\nanswer: true", True),
    ("TRUE", True),
    (" no ", False),
    ("correct.", True),
    ("true", True),
]
UNPARSED_RAW_OUTPUTS = [
    "",
    "The answer is right.",
    '{"score_T1": 1}',
    '{"answer": 1}',
    "[true]",
]


def _judge_row(sample_id: str, dataset: str, answer: str, metadata: dict = None) -> dict:
    return {
        "id": sample_id,
        "image": None,
        "question": "<image 1>\nWhat value is shown?",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": answer,
        "all_answers": [answer],
        "eval_mode": "judge",
        "judge_prompt": TWO_LINE_JUDGE_PROMPT,
        "is_circular": False,
        "source_info": {"dataset": dataset},
        "eval_metrics": {},
        "metadata": json.dumps(metadata or {}),
    }


def _countbench_row(sample_id: str = "countbench_direct", eval_mode: str = "direct") -> dict:
    return {
        "id": sample_id,
        "image": None,
        "question": "<image 1>\nHow many objects are there?",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": "3",
        "all_answers": ["3"],
        "eval_mode": eval_mode,
        "judge_prompt": TWO_LINE_JUDGE_PROMPT if eval_mode == "judge" else "",
        "is_circular": False,
        "source_info": {"dataset": "countbench"},
        "eval_metrics": {},
        "metadata": json.dumps({}),
    }


def _report(rows, capability, vlm_outputs, judge_responses=None, **kwargs):
    evaluator = DatBenchEvaluator(rows, capability=capability)
    vlm_responses = [
        VLMResponse(id=sample_id, raw_output=output)
        for sample_id, output in vlm_outputs.items()
    ]
    return evaluator.compute_metrics(vlm_responses, judge_responses=judge_responses, **kwargs)


def _single_judge_score(judge_response, dataset: str = "mathverse_wo"):
    row = _judge_row("judge-1", dataset, "42")
    judge_responses = None if judge_response is None else [judge_response]
    report = _report([row], "math", {"judge-1": "\\boxed{42}"}, judge_responses)
    return report.results[0]


class JudgeModeScoringTest(unittest.TestCase):
    def test_judge_rows_bypass_dataset_scorer(self) -> None:
        # mathverse_wo has no direct scorer (score_sample raises); ai2d's direct scorer
        # compares choice indices and cannot score a text answer
        self.assertEqual(
            _single_judge_score(JudgeResponse(id="judge-1", raw_judge_output="", verdict="correct")).score,
            1.0,
        )
        self.assertEqual(
            _single_judge_score(JudgeResponse(id="judge-1", raw_judge_output="", verdict="incorrect")).score,
            0.0,
        )

        row = _judge_row("judge-ai2d", "ai2d", "The Cardiac Notch", {"answer_choices": ["Lung", "The Cardiac Notch"]})
        report = _report(
            [row],
            "table",
            {"judge-ai2d": "\\boxed{the cardiac notch}"},
            [JudgeResponse(id="judge-ai2d", raw_judge_output='{"answer": true}', verdict="correct")],
        )
        self.assertEqual(report.results[0].score, 1.0)
        self.assertTrue(report.results[0].is_correct)
        self.assertEqual(report.summary["overall_accuracy"], 1.0)

    def test_verdict_field_takes_precedence_over_raw_output(self) -> None:
        cases = [
            ("incorrect", '{"answer": true}', 0.0),
            ("correct", '{"answer": false}', 1.0),
            ("TRUE", "", 1.0),
            ("no", "", 0.0),
            ("maybe", "answer: true", 1.0),  # unrecognized verdict falls through to the raw text
        ]
        for verdict, raw, expected in cases:
            with self.subTest(verdict=verdict, raw=raw):
                result = _single_judge_score(
                    JudgeResponse(id="judge-1", raw_judge_output=raw, verdict=verdict)
                )
                self.assertEqual(result.score, expected)

    def test_raw_output_formats(self) -> None:
        for raw, expected_match in RAW_OUTPUT_CASES:
            with self.subTest(raw=raw):
                result = _single_judge_score(JudgeResponse(id="judge-1", raw_judge_output=raw))
                self.assertEqual(result.score, 1.0 if expected_match else 0.0)
                score_details = result.metadata["score_details"]
                self.assertEqual(score_details["match"], expected_match)
                self.assertEqual(
                    score_details["judge_verdict"], "correct" if expected_match else "incorrect"
                )
                self.assertEqual(score_details["judge_raw_output"], raw)

    def test_unparsed_raw_output_scores_zero_and_is_counted(self) -> None:
        rows = [
            _judge_row(f"judge-{i}", "mathverse_wo", "42")
            for i in range(len(UNPARSED_RAW_OUTPUTS))
        ]
        vlm_outputs = {row["id"]: "\\boxed{42}" for row in rows}
        judge_responses = [
            JudgeResponse(id=row["id"], raw_judge_output=raw)
            for row, raw in zip(rows, UNPARSED_RAW_OUTPUTS)
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = _report(rows, "math", vlm_outputs, judge_responses)
        self.assertEqual(caught, [])

        for result, raw in zip(report.results, UNPARSED_RAW_OUTPUTS):
            with self.subTest(raw=raw):
                self.assertEqual(result.score, 0.0)
                self.assertFalse(result.is_correct)
                score_details = result.metadata["score_details"]
                self.assertEqual(score_details["judge_verdict"], "unparsed")
                self.assertFalse(score_details["match"])
                self.assertFalse(score_details["judge_missing_response"])
                self.assertEqual(score_details["judge_raw_output"], raw)
        self.assertEqual(report.summary["judge_unparsed_verdicts"], len(UNPARSED_RAW_OUTPUTS))
        self.assertEqual(report.summary["judge_missing_responses"], 0)
        self.assertEqual(report.summary["judge_samples"], len(UNPARSED_RAW_OUTPUTS))
        self.assertEqual(report.summary["judge_accuracy"], 0.0)

    def test_missing_judge_response_scores_zero_non_strict(self) -> None:
        row = _judge_row("judge-1", "mathverse_wo", "42")

        with self.assertWarns(UserWarning):
            report = _report([row], "math", {"judge-1": "\\boxed{42}"}, judge_responses=None)

        result = report.results[0]
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.is_correct)
        self.assertEqual(
            result.metadata["score_details"],
            {
                "score": 0.0,
                "eval_mode": "judge",
                "match": False,
                "judge_verdict": "missing",
                "judge_raw_output": "",
                "judge_missing_response": True,
            },
        )
        summary = report.summary
        self.assertEqual(summary["total_samples"], 1)
        self.assertEqual(summary["total_correct"], 0)
        self.assertEqual(summary["overall_accuracy"], 0.0)
        self.assertEqual(summary["judge_samples"], 1)
        self.assertEqual(summary["judge_missing_responses"], 1)
        self.assertEqual(summary["judge_unparsed_verdicts"], 0)
        self.assertEqual(summary["judge_accuracy"], 0.0)
        self.assertEqual(summary["direct_samples"], 0)
        self.assertEqual(summary["direct_accuracy"], 0.0)
        self.assertEqual(
            summary["dataset_metrics"]["mathverse_wo"],
            {
                "judge_accuracy": 0.0,
                "judge_samples": 1,
                "judge_missing_responses": 1,
                "judge_unparsed_verdicts": 0,
            },
        )

    def test_partially_missing_judge_responses_do_not_warn(self) -> None:
        rows = [_judge_row("judge-1", "mathverse_wo", "42"), _judge_row("judge-2", "mathverse_wo", "7")]
        vlm_outputs = {"judge-1": "\\boxed{42}", "judge-2": "\\boxed{8}"}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = _report(
                rows,
                "math",
                vlm_outputs,
                [JudgeResponse(id="judge-1", raw_judge_output='{"answer": true}')],
            )

        self.assertEqual(caught, [])
        self.assertEqual(report.summary["judge_samples"], 2)
        self.assertEqual(report.summary["judge_missing_responses"], 1)
        self.assertEqual(report.summary["judge_accuracy"], 0.5)
        self.assertEqual(report.summary["overall_accuracy"], 0.5)

    def test_strict_judge_raises(self) -> None:
        row = _judge_row("judge-1", "mathverse_wo", "42")
        vlm_outputs = {"judge-1": "\\boxed{42}"}

        with self.assertRaises(ValueError) as missing:
            _report([row], "math", vlm_outputs, judge_responses=None, strict_judge=True)
        self.assertIn("judge-1", str(missing.exception))

        with self.assertRaises(ValueError) as unparsed:
            _report(
                [row],
                "math",
                vlm_outputs,
                [JudgeResponse(id="judge-1", raw_judge_output="garbage")],
                strict_judge=True,
            )
        self.assertIn("judge-1", str(unparsed.exception))

        report = _report(
            [row],
            "math",
            vlm_outputs,
            [JudgeResponse(id="judge-1", raw_judge_output="", verdict="correct")],
            strict_judge=True,
        )
        self.assertEqual(report.summary["overall_accuracy"], 1.0)

    def test_mixed_direct_and_judge_rows_in_same_capability(self) -> None:
        rows = [_countbench_row(), _judge_row("judge-1", "mathverse_wo", "42")]
        report = _report(
            rows,
            "counting",
            {"countbench_direct": "3", "judge-1": "\\boxed{41}"},
            [JudgeResponse(id="judge-1", raw_judge_output='{"answer": false}', verdict="incorrect")],
        )

        summary = report.summary
        self.assertEqual(summary["overall_accuracy"], 0.5)
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["total_correct"], 1)
        self.assertEqual(summary["capability_accuracy"], {"counting": 0.5})
        self.assertEqual(summary["capability_counts"], {"counting": 2})
        self.assertEqual(summary["direct_samples"], 1)
        self.assertEqual(summary["judge_samples"], 1)
        self.assertEqual(summary["direct_accuracy"], 1.0)
        self.assertEqual(summary["judge_accuracy"], 0.0)
        self.assertEqual(summary["judge_missing_responses"], 0)
        self.assertEqual(summary["judge_unparsed_verdicts"], 0)

        countbench_metrics = summary["dataset_metrics"]["countbench"]
        self.assertEqual(countbench_metrics["accuracy__CountBench-Accuracy"], 1.0)
        self.assertEqual(set(countbench_metrics) & JUDGE_DATASET_METRIC_KEYS, set())
        self.assertEqual(
            summary["dataset_metrics"]["mathverse_wo"],
            {
                "judge_accuracy": 0.0,
                "judge_samples": 1,
                "judge_missing_responses": 0,
                "judge_unparsed_verdicts": 0,
            },
        )

    def test_judge_rows_do_not_enter_dataset_compute_metrics(self) -> None:
        rows = [
            _judge_row("judge-1", "ai2d", "Lung", {"answer_choices": ["Lung", "Heart"]}),
            _judge_row("judge-2", "ai2d", "Heart", {"answer_choices": ["Lung", "Heart"]}),
        ]
        report = _report(
            rows,
            "table",
            {"judge-1": "\\boxed{Lung}", "judge-2": "\\boxed{Lung}"},
            [
                JudgeResponse(id="judge-1", raw_judge_output='{"answer": true}'),
                JudgeResponse(id="judge-2", raw_judge_output='{"answer": false}'),
            ],
        )
        # ai2d's compute_metrics would emit 'accuracy'; judge rows must not reach it
        self.assertEqual(
            report.summary["dataset_metrics"]["ai2d"],
            {
                "judge_accuracy": 0.5,
                "judge_samples": 2,
                "judge_missing_responses": 0,
                "judge_unparsed_verdicts": 0,
            },
        )

        rows = [_countbench_row("countbench_direct"), _countbench_row("countbench_judge", eval_mode="judge")]
        report = _report(
            rows,
            "counting",
            {"countbench_direct": "3", "countbench_judge": "three"},
            [JudgeResponse(id="countbench_judge", raw_judge_output='{"answer": false}')],
        )
        countbench_metrics = report.summary["dataset_metrics"]["countbench"]
        # scorer keys computed over the direct row only (a judge row carries no parsed_answer)
        self.assertEqual(countbench_metrics["accuracy__CountBench-Accuracy"], 1.0)
        self.assertEqual(countbench_metrics["accuracy__CountBench-ParseRate"], 1.0)
        self.assertEqual(countbench_metrics["judge_accuracy"], 0.0)
        self.assertEqual(countbench_metrics["judge_samples"], 1)
        self.assertEqual(countbench_metrics["judge_missing_responses"], 0)
        self.assertEqual(countbench_metrics["judge_unparsed_verdicts"], 0)
        self.assertEqual(report.summary["direct_samples"], 1)
        self.assertEqual(report.summary["judge_samples"], 1)

    def test_judge_sample_score_fields(self) -> None:
        row = _judge_row("judge-1", "mathverse_wo", "42")
        report = _report(
            [row],
            "math",
            {"judge-1": "The value is \\boxed{42}."},
            [JudgeResponse(id="judge-1", raw_judge_output='{"answer": true}', verdict="correct")],
        )

        result = report.results[0]
        self.assertEqual(result.id, "judge-1")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.vlm_output, "The value is \\boxed{42}.")
        self.assertEqual(result.ground_truth, "42")
        self.assertEqual(set(result.metadata), SAMPLE_SCORE_METADATA_KEYS)
        self.assertEqual(result.metadata["eval_mode"], "judge")
        self.assertEqual(result.metadata["dataset"], "mathverse_wo")
        self.assertEqual(result.metadata["capability"], "math")
        score_details = result.metadata["score_details"]
        self.assertEqual(set(score_details), JUDGE_SCORE_DETAIL_KEYS)
        self.assertEqual(score_details["eval_mode"], "judge")
        self.assertTrue(score_details["match"])
        self.assertEqual(score_details["judge_verdict"], "correct")
        self.assertEqual(score_details["judge_raw_output"], '{"answer": true}')
        self.assertFalse(score_details["judge_missing_response"])

    def test_judge_response_for_direct_row_is_ignored(self) -> None:
        report = _report(
            [_countbench_row()],
            "counting",
            {"countbench_direct": "3"},
            [
                JudgeResponse(id="countbench_direct", raw_judge_output="", verdict="incorrect"),
                JudgeResponse(id="unknown-id", raw_judge_output="", verdict="incorrect"),
            ],
        )

        self.assertEqual(report.results[0].score, 1.0)
        self.assertEqual(report.results[0].metadata["score_details"]["parsed_answer"], 3)
        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        self.assertEqual(report.summary["direct_samples"], 1)
        self.assertEqual(report.summary["judge_samples"], 0)
        self.assertEqual(report.summary["judge_missing_responses"], 0)
        self.assertEqual(report.summary["judge_accuracy"], 0.0)

    def test_judge_row_without_vlm_response_is_skipped(self) -> None:
        row = _judge_row("judge-1", "mathverse_wo", "42")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = _report([row], "math", {}, judge_responses=None)

        self.assertEqual(caught, [])
        self.assertEqual(report.results, [])
        self.assertEqual(report.summary["total_samples"], 0)
        self.assertEqual(report.summary["judge_samples"], 0)
        self.assertEqual(report.summary["judge_missing_responses"], 0)
        self.assertEqual(report.summary["dataset_metrics"], {})

        # strict mode only checks judge rows that have a VLM response
        _report([row], "math", {}, judge_responses=None, strict_judge=True)


class ParseJudgeVerdictTest(unittest.TestCase):
    def test_raw_output_corpus(self) -> None:
        for raw, expected in RAW_OUTPUT_CASES:
            with self.subTest(raw=raw):
                self.assertIs(parse_judge_verdict(None, raw), expected)
        for raw in UNPARSED_RAW_OUTPUTS:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_judge_verdict(None, raw))
        self.assertIsNone(parse_judge_verdict(None, None))

    def test_verdict_field(self) -> None:
        self.assertIs(parse_judge_verdict("correct", None), True)
        self.assertIs(parse_judge_verdict(" Incorrect ", None), False)
        self.assertIs(parse_judge_verdict("yes", None), True)
        self.assertIs(parse_judge_verdict("FALSE", None), False)
        self.assertIsNone(parse_judge_verdict("maybe", None))
        self.assertIsNone(parse_judge_verdict("", None))
        self.assertIs(parse_judge_verdict("incorrect", '{"answer": true}'), False)
        self.assertIs(parse_judge_verdict("maybe", '{"answer": true}'), True)


class JudgeRequestTest(unittest.TestCase):
    def _sample(self, judge_prompt: str) -> DatBenchSample:
        row = _judge_row("judge-1", "mathverse_wo", "42")
        row["judge_prompt"] = judge_prompt
        return DatBenchSample.from_hf_row(row, "math")

    def test_judge_request_rewrites_two_line_instruction(self) -> None:
        request = JudgeRequest.from_sample_and_response(
            self._sample(TWO_LINE_JUDGE_PROMPT),
            VLMResponse(id="judge-1", raw_output="\\boxed{42}"),
        )

        rubric = TWO_LINE_JUDGE_PROMPT.split("\n\nRespond with exactly two lines:")[0]
        self.assertEqual(request.judge_prompt, rubric + SINGLE_TOKEN_SUFFIX)
        self.assertEqual(request.id, "judge-1")
        self.assertEqual(request.ground_truth, "42")
        self.assertEqual(request.vlm_response, "\\boxed{42}")

    def test_judge_request_keeps_other_prompts_and_prefers_parsed_answer(self) -> None:
        prompt = 'Return only JSON with this schema: {"answer": true} or {"answer": false}'
        request = JudgeRequest.from_sample_and_response(
            self._sample(prompt),
            VLMResponse(id="judge-1", raw_output="The answer is 42.", parsed_answer="42"),
        )

        self.assertEqual(request.judge_prompt, prompt)
        self.assertEqual(request.vlm_response, "42")


if __name__ == "__main__":
    unittest.main()
