"""Tests for OCRBench v2 text spotting scoring."""

import unittest

from datbench.datasets_scoring import ocrbench_v2

# Two words, each given as an 8-value polygon like the reference data.
SAMPLE = {
    "type": "text spotting en",
    "bbox_list": [
        [10, 10, 110, 10, 110, 60, 10, 60],
        [200, 200, 300, 200, 300, 260, 200, 260],
    ],
    "content": ["HELLO", "WORLD"],
    "answers": [
        "10,10,110,10,110,60,10,60,HELLO\n200,200,300,200,300,260,200,260,WORLD"
    ],
}


def _score(prediction: str, sample: dict = SAMPLE) -> float:
    return ocrbench_v2._text_spotting_evaluation(prediction, sample)


class TextSpottingTest(unittest.TestCase):
    def test_perfect_prediction_scores_one(self) -> None:
        pred = "[(10, 10, 110, 60, 'HELLO'), (200, 200, 300, 260, 'WORLD')]"
        self.assertEqual(_score(pred), 1.0)

    def test_score_sample_routes_to_spotting_metric(self) -> None:
        pred = "[(10, 10, 110, 60, 'HELLO'), (200, 200, 300, 260, 'WORLD')]"
        self.assertEqual(ocrbench_v2.score_sample(SAMPLE, pred)["score"], 1.0)

    def test_boxed_prediction_scores_one(self) -> None:
        pred = "\\boxed{[(10, 10, 110, 60, 'HELLO'), (200, 200, 300, 260, 'WORLD')]}"
        self.assertEqual(_score(pred), 1.0)

    def test_boxed_prediction_keeps_escaped_quotes_intact(self) -> None:
        sample = dict(SAMPLE, content=["ARBY'S", "WORLD"])
        pred = "\\boxed{[(10, 10, 110, 60, 'ARBY\\'S'), (200, 200, 300, 260, 'WORLD')]}"
        self.assertEqual(_score(pred, sample), 1.0)

    def test_text_match_ignores_case(self) -> None:
        pred = "[(10, 10, 110, 60, 'hello'), (200, 200, 300, 260, 'world')]"
        self.assertEqual(_score(pred), 1.0)

    def test_one_of_two_words_scores_hmean(self) -> None:
        # precision 1/1, recall 1/2
        pred = "[(10, 10, 110, 60, 'HELLO')]"
        self.assertAlmostEqual(_score(pred), 2 / 3)

    def test_boxes_in_wrong_place_score_zero(self) -> None:
        pred = "[(500, 500, 600, 600, 'HELLO'), (700, 700, 800, 800, 'WORLD')]"
        self.assertEqual(_score(pred), 0.0)

    def test_wrong_text_scores_zero(self) -> None:
        pred = "[(10, 10, 110, 60, 'GOODBYE'), (200, 200, 300, 260, 'MOON')]"
        self.assertEqual(_score(pred), 0.0)

    def test_dont_care_regions_are_ignored(self) -> None:
        sample = dict(SAMPLE, content=["HELLO", "###"])
        pred = "[(10, 10, 110, 60, 'HELLO')]"
        self.assertEqual(_score(pred, sample), 1.0)

    def test_unparseable_prediction_scores_zero(self) -> None:
        self.assertEqual(_score("I cannot read this image."), 0.0)


if __name__ == "__main__":
    unittest.main()
