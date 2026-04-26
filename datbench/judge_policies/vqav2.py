"""VQA-V2 semantic judge prompt policy for DatBench release tooling."""

from __future__ import annotations

from datbench.schema import DatBenchSample

VQA_V2_JUDGE_POLICY_MARKER = "VQA-V2 semantic judge policy v2"
LEGACY_VQA_V2_POLICY_MARKERS = (
    "Final-answer extraction policy for VQA-V2 semantic judging",
)

VQA_V2_JUDGE_FINAL_ANSWER_POLICY = r"""

VQA-V2 semantic judge policy v2:
- Judge the model's final answer content, not wrapper syntax.
- If the model response contains content inside \boxed{...}, treat that content as the model answer.
- If the model response contains an `Answer:` or `Final Answer:` line, treat the text after that marker as the model answer.
- Ignore harmless wrapper syntax, capitalization, and surrounding whitespace when applying the semantic rubric.
- Use the full set of human answers as acceptable references. The primary `ground_truth` value is not exhaustive.
- Accept compatible refinements and hyponyms when they do not contradict the question or human answers, such as "seagull" for "bird" or "wine bottle" for "bottle".
- Accept short descriptive phrases that clearly contain the same answer and add only compatible detail, such as "directions to cities and distances" for "directions" or "on the wall to the right of the toilet" for "on wall".
- Accept approximate numeric descriptions for qualitative answers when they are consistent with the human answer set, such as "1 foot" for "shallow", unless the question asks for an exact count or exact number.
- For OCR text and proper nouns, ignore casing, punctuation, and minor spacing, but do not accept different text.
- Be strict for yes/no polarity, exact counts or exact numbers, incompatible colors, incompatible object identity, incompatible spatial relations, and extra details that contradict the references.
- If human answers disagree, accept any plausible answer represented by at least one non-outlier human answer; do not require the model answer to match the majority wording.
- If the model gives multiple conflicting answers, judge it incorrect unless the final answer is unambiguous and correct.
- Return only JSON with this schema:
{"answer": true} or {"answer": false}
""".strip()


def _strip_existing_vqa_v2_policy(judge_prompt: str) -> str:
    policy_start = len(judge_prompt)
    for marker in (VQA_V2_JUDGE_POLICY_MARKER, *LEGACY_VQA_V2_POLICY_MARKERS):
        marker_index = judge_prompt.find(marker)
        if marker_index != -1:
            policy_start = min(policy_start, marker_index)

    if policy_start == len(judge_prompt):
        return judge_prompt.rstrip()
    return judge_prompt[:policy_start].rstrip()


def is_vqa_v2_source(sample: DatBenchSample) -> bool:
    return sample.source_info.get("dataset") == "vqa-v2"


def with_vqa_v2_final_answer_policy(judge_prompt: str) -> str:
    base_prompt = _strip_existing_vqa_v2_policy(judge_prompt)
    if not base_prompt:
        return VQA_V2_JUDGE_FINAL_ANSWER_POLICY
    return f"{base_prompt}\n\n{VQA_V2_JUDGE_FINAL_ANSWER_POLICY}".strip()


def vqa_v2_judge_prompt_for_sample(sample: DatBenchSample) -> str:
    judge_prompt = sample.judge_prompt or ""
    if not is_vqa_v2_source(sample):
        return judge_prompt
    return with_vqa_v2_final_answer_policy(judge_prompt)
