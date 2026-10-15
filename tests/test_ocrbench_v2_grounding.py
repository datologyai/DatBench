"""Tests for OCRBench v2 text grounding and VQA-with-position scoring."""

import unittest

from datbench.datasets_scoring import ocrbench_v2

# Text grounding rows carry the reference box as the four answer values.
GROUNDING = {
    "type": "text grounding en",
    "answers": ["100", "100", "200", "200"],
    "bbox": None,
}


def _score(prediction: str, sample: dict = GROUNDING) -> float:
    return ocrbench_v2.score_sample(sample, prediction)["score"]


class TextGroundingTest(unittest.TestCase):
    def test_exact_box_scores_one(self) -> None:
        self.assertEqual(_score("(100, 100, 200, 200)"), 1.0)

    def test_boxed_box_scores_one(self) -> None:
        self.assertEqual(_score("\\boxed{(100, 100, 200, 200)}"), 1.0)

    def test_partial_overlap_scores_iou(self) -> None:
        # Top half of the reference box.
        self.assertAlmostEqual(_score("(100, 100, 200, 150)"), 0.5)

    def test_close_box_without_shared_coordinates_scores_iou(self) -> None:
        self.assertAlmostEqual(_score("(101, 101, 199, 199)"), 0.9604)

    def test_disjoint_box_scores_zero(self) -> None:
        self.assertEqual(_score("(300, 300, 400, 400)"), 0.0)

    def test_disjoint_box_sharing_a_coordinate_scores_zero(self) -> None:
        self.assertEqual(_score("(200, 300, 300, 400)"), 0.0)

    def test_missing_box_scores_zero(self) -> None:
        self.assertEqual(_score("The text is near the top."), 0.0)


class VqaWithPositionTest(unittest.TestCase):
    SAMPLE = {
        "type": "VQA with position en",
        "answers": ["Walter"],
        "bbox": [100, 100, 200, 200],
    }

    def test_answer_and_box_each_score_half(self) -> None:
        pred = '{"answer": "Walter", "bbox": "(100, 100, 200, 200)"}'
        self.assertEqual(_score(pred, self.SAMPLE), 1.0)
        pred = '{"answer": "Walter", "bbox": "(300, 300, 400, 400)"}'
        self.assertEqual(_score(pred, self.SAMPLE), 0.5)


if __name__ == "__main__":
    unittest.main()
