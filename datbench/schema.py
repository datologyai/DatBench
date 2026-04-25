"""DatBench evaluation dataclasses.

These define the handoff between the evaluation harness and user's inference code.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass
class DatBenchSample:
    """Single sample from DatBench dataset.

    Represents one evaluation instance with image, question, and ground truth.
    """
    id: str
    capability: str
    image: Any  # PIL.Image.Image
    question: str
    prompt_format: Dict[str, str]
    answer: str
    all_answers: List[str] = field(default_factory=list)
    eval_mode: str = "direct"  # 'direct' or 'judge'
    judge_prompt: str = ""
    is_circular: bool = False
    metadata: str = "{}"  # JSON string with dataset-specific metadata
    source_info: Dict[str, str] = field(default_factory=dict)
    eval_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hf_row(cls, row: Dict[str, Any], capability: str) -> "DatBenchSample":
        """Convert a HF dataset row to DatBenchSample.

        Args:
            row: HuggingFace dataset row (dict)
            capability: Capability name from dataset subset

        Returns:
            DatBenchSample instance
        """
        return cls(
            id=row["id"],
            capability=capability,
            image=row["image"],
            question=row["question"],
            prompt_format=row.get("prompt_format", {"prefix": "", "suffix": ""}),
            answer=row["answer"],
            all_answers=row.get("all_answers", [row["answer"]]),
            eval_mode=row.get("eval_mode", "direct"),
            judge_prompt=row.get("judge_prompt", ""),
            is_circular=row.get("is_circular", False),
            metadata=row.get("metadata", "{}"),
            source_info=row.get("source_info", {}),
            eval_metrics=row.get("eval_metrics", {}),
        )


@dataclass
class InferenceTask:
    """Task for VLM inference.

    Contains the image and question prompt for model evaluation.
    """
    id: str
    image: Any  # PIL.Image.Image
    question: str
    eval_mode: str  # 'direct' or 'judge'

    @classmethod
    def from_sample(cls, sample: DatBenchSample) -> "InferenceTask":
        """Create an InferenceTask from a DatBenchSample."""
        # Format full prompt: prefix + question + suffix
        prefix = sample.prompt_format.get('prefix', '')
        suffix = sample.prompt_format.get('suffix', '')
        formatted_question = f"{prefix}{sample.question}{suffix}"

        return cls(
            id=sample.id,
            image=sample.image,
            question=formatted_question,
            eval_mode=sample.eval_mode,
        )


@dataclass
class VLMResponse:
    """VLM model output for a single sample."""
    id: str
    raw_output: str
    parsed_answer: Optional[str] = None


VQA_V2_JUDGE_FINAL_ANSWER_POLICY = r"""

Final-answer extraction policy for VQA-V2 semantic judging:
- Judge the model's final answer content, not wrapper syntax.
- If the model response contains content inside \boxed{...}, treat that content as the model answer.
- If the model response contains an `Answer:` or `Final Answer:` line, treat the text after that marker as the model answer.
- Ignore harmless wrapper syntax, capitalization, and surrounding whitespace when applying the semantic rubric.
- Keep the semantic comparison strict for answer content: wrong object identity, color, count, yes/no polarity, text/OCR, or spatial relation remains incorrect.
""".strip()


def _with_vqa_v2_final_answer_policy(sample: "DatBenchSample", judge_prompt: str) -> str:
    if sample.source_info.get("dataset") != "vqa-v2":
        return judge_prompt
    if "Final-answer extraction policy for VQA-V2 semantic judging" in judge_prompt:
        return judge_prompt
    return f"{judge_prompt.rstrip()}\n\n{VQA_V2_JUDGE_FINAL_ANSWER_POLICY}".strip()


@dataclass
class JudgeRequest:
    """Request for judge-based evaluation.

    Contains VLM response and judge prompt for LM-based scoring.
    """
    id: str
    vlm_response: str
    ground_truth: str
    judge_prompt: str

    @classmethod
    def from_sample_and_response(
        cls,
        sample: DatBenchSample,
        vlm_response: VLMResponse
    ) -> "JudgeRequest":
        """Create a JudgeRequest from a sample and VLM response."""
        judge_prompt = sample.judge_prompt or ""
        # DatBench HF rows sometimes require a 2-line output:
        #   Line 1: Reason: ...
        #   Line 2: answer: true/false
        # With low max_new_tokens this can truncate before the verdict. We only
        # need the verdict for scoring, so rewrite the output format to a single
        # verdict line while preserving all rubric content above.
        marker = "Respond with exactly two lines:"
        idx = judge_prompt.rfind(marker)
        if idx != -1:
            judge_prompt = (
                judge_prompt[:idx].rstrip()
                + "\n\nOutput exactly one boolean token: true or false. answer: "
            )
        judge_prompt = _with_vqa_v2_final_answer_policy(sample, judge_prompt)
        return cls(
            id=sample.id,
            vlm_response=vlm_response.parsed_answer or vlm_response.raw_output,
            ground_truth=sample.answer,
            judge_prompt=judge_prompt,
        )


@dataclass
class JudgeResponse:
    """Judge model output for a single sample."""
    id: str
    raw_judge_output: str
    verdict: Optional[str] = None


@dataclass
class SampleScore:
    """Per-sample evaluation result."""
    id: str
    score: float
    vlm_output: str
    ground_truth: str
    is_correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatBenchReport:
    """Evaluation report with summary metrics and per-sample results."""
    summary: Dict[str, Any]
    results: List[SampleScore]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
