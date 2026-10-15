"""Tests for Pixmo Points Eval mask-edge handling."""

import json
import unittest

from datbench import DatBenchEvaluator, VLMResponse
from datbench.datasets_scoring import pixmo_points_eval


def _mask_with_true_at(height: int, width: int, row: int, col: int) -> list:
    mask = [[False] * width for _ in range(height)]
    mask[row][col] = True
    return mask


def _pixmo_row(points: list, masks: list) -> dict:
    metadata = {"points_target": points, "points_mask": masks, "has_target": True}
    return {
        "id": "pixmo_sample",
        "image": None,
        "question": "<image 1>\nWhere is the marker?",
        "prompt_format": {"prefix": "", "suffix": "\nAnswer:"},
        "answer": json.dumps(points),
        "all_answers": [json.dumps(points)],
        "eval_mode": "direct",
        "judge_prompt": "",
        "is_circular": False,
        "source_info": {"dataset": "pixmo_points_eval"},
        "eval_metrics": {},
        "metadata": json.dumps(metadata),
    }


class PointInMaskTest(unittest.TestCase):
    def test_right_edge_reads_last_column(self) -> None:
        mask = _mask_with_true_at(height=2, width=3, row=1, col=2)
        self.assertTrue(pixmo_points_eval._point_in_mask((1.0, 0.5), mask))
        self.assertFalse(pixmo_points_eval._point_in_mask((0.5, 0.5), mask))

    def test_bottom_edge_reads_last_row(self) -> None:
        mask = _mask_with_true_at(height=3, width=2, row=2, col=0)
        self.assertTrue(pixmo_points_eval._point_in_mask((0.0, 1.0), mask))
        self.assertFalse(pixmo_points_eval._point_in_mask((0.0, 0.0), mask))

    def test_bottom_right_corner(self) -> None:
        mask = _mask_with_true_at(height=4, width=4, row=3, col=3)
        self.assertTrue(pixmo_points_eval._point_in_mask((1.0, 1.0), mask))

    def test_interior_point_unchanged(self) -> None:
        mask = _mask_with_true_at(height=4, width=4, row=1, col=2)
        self.assertTrue(pixmo_points_eval._point_in_mask((0.6, 0.3), mask))
        self.assertFalse(pixmo_points_eval._point_in_mask((0.3, 0.6), mask))

    def test_empty_mask_is_never_hit(self) -> None:
        self.assertFalse(pixmo_points_eval._point_in_mask((0.5, 0.5), []))
        self.assertFalse(pixmo_points_eval._point_in_mask((0.5, 0.5), [[]]))


class PixmoScoreSampleTest(unittest.TestCase):
    def test_prediction_on_bottom_right_edge_hits_last_cell(self) -> None:
        mask = _mask_with_true_at(height=4, width=4, row=3, col=3)
        sample = {
            "points": [{"x": 100.0, "y": 100.0}],
            "masks": [mask],
            "has_target": True,
        }

        details = pixmo_points_eval.score_sample(sample, "\\boxed{1000, 1000}")

        self.assertEqual(details["predicted_points"], [(1.0, 1.0)])
        self.assertEqual(details["precision"], 1.0)
        self.assertEqual(details["recall"], 1.0)
        self.assertEqual(details["score"], 1.0)

    def test_evaluator_end_to_end(self) -> None:
        mask = _mask_with_true_at(height=4, width=4, row=3, col=3)
        row = _pixmo_row(points=[{"x": 100.0, "y": 100.0}], masks=[mask])
        evaluator = DatBenchEvaluator([row], capability="grounding")

        report = evaluator.compute_metrics(
            [VLMResponse(id="pixmo_sample", raw_output="\\boxed{1000, 1000}")]
        )

        self.assertEqual(report.summary["overall_accuracy"], 1.0)
        score_details = report.results[0].metadata["score_details"]
        self.assertEqual(score_details["precision"], 1.0)
        self.assertEqual(score_details["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
