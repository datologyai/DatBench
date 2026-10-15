"""Tests for final-answer extraction from boxed and free-form model outputs."""

import unittest

from datbench.datasets_scoring.evaluation_utils.erma_utils import extract_final_answer


class FinalAnswerExtractionTest(unittest.TestCase):
    def test_nested_braces_and_trailing_terms_are_preserved(self) -> None:
        cases = [
            (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            (r"\boxed{-\frac{21}{29}}", r"-\frac{21}{29}"),
            (r"\boxed{4\sqrt{3}}", r"4\sqrt{3}"),
            (
                r"\boxed{x^{3}+12 x^{2}+48 x+64}",
                r"x^{3}+12 x^{2}+48 x+64",
            ),
            (r"\boxed{y=\sqrt{49-x^{2}}+7}", r"y=\sqrt{49-x^{2}}+7"),
            (r"\boxed{904.78 \text{ cm}^3}", r"904.78 \text{ cm}^3"),
            (
                r"\boxed{\frac{2\pi}{\frac{\pi}{2}}}",
                r"\frac{2\pi}{\frac{\pi}{2}}",
            ),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), expected)

    def test_last_complete_box_wins(self) -> None:
        cases = [
            (r"\boxed{A}, then \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            (r"\boxed{\frac{1}{2}}, then \boxed{B}", "B"),
            (r"\boxed{42}, then \boxed{\frac{1}{2}", "42"),
            (r"\boxed{42}, then \boxed{}", "42"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), expected)

    def test_incomplete_box_uses_existing_answer_or_line_fallback(self) -> None:
        cases = [
            ("\\boxed{\\frac{1}{2}\nANSWER: Paris", "Paris"),
            (r"\boxed{\frac{1}{2}", r"\boxed{\frac{1}{2}"),
            ("Reasoning\nFinal answer: 42", "42"),
            ("Reasoning\n'Paris'", "Paris"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), expected)

    def test_escaped_literal_braces_do_not_close_or_open_groups(self) -> None:
        cases = [
            (r"\boxed{\{1, 2\}}", r"\{1, 2\}"),
            (r"\boxed{\text{a\}b}}", r"\text{a\}b}"),
            (r"\boxed{\text{a\{b}}", r"\text{a\{b}"),
            (r"\boxed{x\\{y}}", r"x\\{y}"),
            (r"\boxed{x\\\}y}", r"x\\\}y"),
            (
                r"\boxed{f(x)=\left\{\begin{array}{l}x, & -1<x<1,\\ "
                r"-x, & 1<x<3.\end{array}\right.}",
                r"f(x)=\left\{\begin{array}{l}x, & -1<x<1,\\ "
                r"-x, & 1<x<3.\end{array}\right.",
            ),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), expected)

    def test_single_and_multiple_backslash_box_markers(self) -> None:
        for marker in [r"\boxed", r"\\boxed", r"\\\boxed"]:
            with self.subTest(marker=marker):
                self.assertEqual(
                    extract_final_answer(marker + r"{\frac{1}{2}}"),
                    r"\frac{1}{2}",
                )

    def test_box_priority_and_candidate_cleaning_are_preserved(self) -> None:
        cases = [
            ("\\boxed{42}\nANSWER: 43", "42"),
            (
                r"\boxed{\frac{1}{2}} <|begin_of_box|>Paris<|end_of_box|>",
                "Paris",
            ),
            (
                "<|begin_of_box|>42<|end_of_box|> <|begin_of_box|>Paris<|end_of_box|>",
                "Paris",
            ),
            (r'\boxed{"<answer> \frac{1}{2} </answer>"}', r"\frac{1}{2}"),
            ("\\boxed{ \n\\frac{1}{2}\n }", r"\frac{1}{2}"),
            (r"\boxed{<answer>}", ""),
            (r"\boxed{ }", ""),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), expected)

    def test_empty_and_non_string_inputs_remain_empty(self) -> None:
        for response in ["", "   ", None, 42]:
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response), "")


if __name__ == "__main__":
    unittest.main()
