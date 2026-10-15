"""Tests for OCRBench v2 category aggregation in compute_metrics."""

import json
import unittest

from datbench import DatBenchEvaluator, VLMResponse
from datbench.datasets_scoring import ocrbench_v2

# Official leaderboard buckets (get_score.py): metric key -> task types it averages.
EN_BUCKETS = {
    "Recognition": ["text recognition en", "fine-grained text recognition en", "full-page OCR en"],
    "Referring": ["text grounding en", "VQA with position en"],
    "Spotting": ["text spotting en"],
    "Extraction": ["key information extraction en", "key information mapping en"],
    "Parsing": ["document parsing en", "chart parsing en", "table parsing en", "formula recognition en"],
    "Calculation": ["math QA en", "text counting en"],
    "Understanding": ["document classification en", "cognition VQA en", "diagram QA en"],
    "Reasoning": ["reasoning VQA en", "science QA en", "APP agent en", "ASCII art classification en"],
}

CN_BUCKETS = {
    "CN_Recognition": ["full-page OCR cn"],
    "CN_Extraction": ["key information extraction cn", "handwritten answer extraction cn"],
    "CN_Parsing": ["document parsing cn", "table parsing cn", "formula recognition cn"],
    "CN_Understanding": ["cognition VQA cn"],
    "CN_Reasoning": ["reasoning VQA cn", "text translation cn"],
}


def _prediction(task_type: str, score: float) -> dict:
    """Build a prediction in the shape DatBenchEvaluator hands to compute_metrics."""
    return {"score_details": {"score": score, "task_type": task_type}}


def _mean(values: list) -> float:
    return sum(values) / len(values)


def _ocrbench_row(sample_id: str, task_type: str, answer: str) -> dict:
    return {
        "id": sample_id,
        "image": None,
        "question": "<image 1>\nTranscribe the formula.",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": answer,
        "all_answers": [answer],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": "ocrbench-v2"},
        "eval_metrics": {},
        "metadata": json.dumps({"type": task_type, "eval": None, "dataset_name": "synthetic"}),
    }


class OCRBenchV2ComputeMetricsTest(unittest.TestCase):
    def test_every_task_type_lands_in_its_official_bucket(self) -> None:
        # Give every task type a distinct score so a row landing in the wrong
        # bucket changes that bucket's mean.
        predictions = []
        expected = {}
        score = 0.0
        for bucket, task_types in {**EN_BUCKETS, **CN_BUCKETS}.items():
            bucket_scores = []
            for task_type in task_types:
                score += 0.05
                predictions.append(_prediction(task_type, score))
                bucket_scores.append(score)
            expected[bucket] = _mean(bucket_scores)

        metrics = ocrbench_v2.compute_metrics(predictions)

        for bucket, value in expected.items():
            self.assertAlmostEqual(metrics[bucket], value, msg=bucket)
        en_average = _mean([expected[b] for b in EN_BUCKETS])
        cn_average = _mean([expected[b] for b in CN_BUCKETS])
        self.assertAlmostEqual(metrics["Average"], en_average)
        self.assertAlmostEqual(metrics["accuracy"], en_average)
        self.assertAlmostEqual(metrics["score"], en_average)
        self.assertAlmostEqual(metrics["CN_Average"], cn_average)

    def test_cn_parsing_tasks_do_not_leak_into_en_parsing(self) -> None:
        predictions = [
            _prediction("formula recognition cn", 1.0),
            _prediction("table parsing cn", 1.0),
            _prediction("document parsing cn", 0.5),
            _prediction("formula recognition en", 0.0),
        ]

        metrics = ocrbench_v2.compute_metrics(predictions)

        self.assertAlmostEqual(metrics["CN_Parsing"], _mean([1.0, 1.0, 0.5]))
        self.assertEqual(metrics["Parsing"], 0.0)
        self.assertEqual(metrics["Average"], 0.0)
        self.assertAlmostEqual(metrics["CN_Average"], metrics["CN_Parsing"] / len(CN_BUCKETS))

    def test_unknown_task_type_is_ignored(self) -> None:
        metrics = ocrbench_v2.compute_metrics([_prediction("not a task", 1.0)])

        for bucket in {**EN_BUCKETS, **CN_BUCKETS}:
            self.assertEqual(metrics[bucket], 0.0, msg=bucket)
        self.assertEqual(metrics["Average"], 0.0)
        self.assertEqual(metrics["CN_Average"], 0.0)

    def test_empty_results(self) -> None:
        self.assertEqual(ocrbench_v2.compute_metrics([]), {"accuracy": 0.0})

    def test_evaluator_reports_cn_formula_under_cn_parsing(self) -> None:
        rows = [
            _ocrbench_row("ocrbench_cn", "formula recognition cn", "x^2+y^2=1"),
            _ocrbench_row("ocrbench_en", "formula recognition en", "a+b=c"),
        ]
        evaluator = DatBenchEvaluator(rows, capability="document")

        report = evaluator.compute_metrics(
            [
                VLMResponse(id="ocrbench_cn", raw_output="x^2+y^2=1"),
                VLMResponse(id="ocrbench_en", raw_output="wrong"),
            ]
        )

        self.assertEqual(report.summary["overall_accuracy"], 0.5)
        dataset_metrics = report.summary["dataset_metrics"]["ocrbench-v2"]
        self.assertEqual(dataset_metrics["CN_Parsing"], 1.0)
        self.assertEqual(dataset_metrics["Parsing"], 0.0)
        self.assertEqual(dataset_metrics["accuracy"], 0.0)
        self.assertAlmostEqual(dataset_metrics["CN_Average"], 1.0 / len(CN_BUCKETS))


if __name__ == "__main__":
    unittest.main()
