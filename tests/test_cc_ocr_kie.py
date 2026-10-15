"""Tests for CC-OCR KIE JSON extraction."""

import unittest

from datbench.datasets_scoring import cc_ocr_kie

EXPECTED = {"COMPANY": "ACME", "TOTAL": "12.50"}


class ExtractJsonTest(unittest.TestCase):
    def test_boxed_braces_shared_with_json(self) -> None:
        text = '\\boxed{"COMPANY": "ACME", "TOTAL": "12.50"}'
        self.assertEqual(cc_ocr_kie._extract_json(text), EXPECTED)

    def test_boxed_braces_shared_with_nested_json(self) -> None:
        text = '\\boxed{"COMPANY": "ACME", "ITEMS": [{"NAME": "pen"}]}'
        self.assertEqual(
            cc_ocr_kie._extract_json(text),
            {"COMPANY": "ACME", "ITEMS": [{"NAME": "pen"}]},
        )

    def test_boxed_json_object(self) -> None:
        text = '\\boxed{{"COMPANY": "ACME", "TOTAL": "12.50"}}'
        self.assertEqual(cc_ocr_kie._extract_json(text), EXPECTED)

    def test_boxed_code_fence(self) -> None:
        text = '\\boxed{```json\n{"COMPANY": "ACME", "TOTAL": "12.50"}\n```}'
        self.assertEqual(cc_ocr_kie._extract_json(text), EXPECTED)

    def test_boxed_text_before_json(self) -> None:
        text = '\\boxed{Answer: {"COMPANY": "ACME", "TOTAL": "12.50"}}'
        self.assertEqual(cc_ocr_kie._extract_json(text), EXPECTED)

    def test_bare_json(self) -> None:
        text = '{"COMPANY": "ACME", "TOTAL": "12.50"}'
        self.assertEqual(cc_ocr_kie._extract_json(text), EXPECTED)

    def test_shared_braces_score_like_bare_json(self) -> None:
        sample = {"answer": '{"COMPANY": "ACME", "TOTAL": "12.50"}'}
        bare = cc_ocr_kie.score_sample(sample, '{"COMPANY": "ACME", "TOTAL": "12.50"}')
        boxed = cc_ocr_kie.score_sample(
            sample, '\\boxed{"COMPANY": "ACME", "TOTAL": "12.50"}'
        )
        self.assertAlmostEqual(bare["score"], 1.0, places=5)
        self.assertEqual(boxed["score"], bare["score"])


if __name__ == "__main__":
    unittest.main()
