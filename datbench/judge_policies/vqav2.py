"""VQA-V2 semantic judge prompt policy for DatBench release tooling."""

from __future__ import annotations

from datbench.schema import DatBenchSample

VQA_V2_JUDGE_FINAL_ANSWER_POLICY = r"""

Final-answer extraction policy for VQA-V2 semantic judging:
- Judge the model's final answer content, not wrapper syntax.
- If the model response contains content inside \boxed{...}, treat that content as the model answer.
- If the model response contains an `Answer:` or `Final Answer:` line, treat the text after that marker as the model answer.
- Ignore harmless wrapper syntax, capitalization, and surrounding whitespace when applying the semantic rubric.
- Keep the semantic comparison strict for answer content: wrong object identity, color, count, yes/no polarity, text/OCR, or spatial relation remains incorrect.
""".strip()


def is_vqa_v2_source(sample: DatBenchSample) -> bool:
    return sample.source_info.get("dataset") == "vqa-v2"


def with_vqa_v2_final_answer_policy(judge_prompt: str) -> str:
    if "Final-answer extraction policy for VQA-V2 semantic judging" in judge_prompt:
        return judge_prompt
    return f"{judge_prompt.rstrip()}\n\n{VQA_V2_JUDGE_FINAL_ANSWER_POLICY}".strip()


def vqa_v2_judge_prompt_for_sample(sample: DatBenchSample) -> str:
    judge_prompt = sample.judge_prompt or ""
    if not is_vqa_v2_source(sample):
        return judge_prompt
    return with_vqa_v2_final_answer_policy(judge_prompt)
