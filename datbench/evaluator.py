"""DatBench evaluation harness."""

import json
import re
import warnings
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from .schema import (
    DatBenchSample,
    InferenceTask,
    VLMResponse,
    JudgeRequest,
    JudgeResponse,
    SampleScore,
    DatBenchReport,
)
from .datasets_scoring import DATASET_SCORING_MODULES
from .datasets_scoring.evaluation_utils.erma_utils import extract_final_answer


COMMON_MCQ_METADATA_FIELDS = ("options", "index2ans", "all_choices")

# Judge verdict normalization (see parse_judge_verdict)
JUDGE_TRUE_TOKENS = frozenset({"true", "yes", "correct"})
JUDGE_FALSE_TOKENS = frozenset({"false", "no", "incorrect"})
JUDGE_ANSWER_LINE_RE = re.compile(r"\banswer\s*:\s*(true|false|yes|no)\b", re.IGNORECASE)
JUDGE_CODE_FENCE_RE = re.compile(r"^```(?:[\w-]+\n)?(.*?)```$", re.DOTALL)

# Labels written to score_details['judge_verdict'] for judge-mode samples
JUDGE_VERDICT_CORRECT = "correct"
JUDGE_VERDICT_INCORRECT = "incorrect"
JUDGE_VERDICT_MISSING = "missing"
JUDGE_VERDICT_UNPARSED = "unparsed"

JUDGE_EVAL_MODE = "judge"
MAX_SAMPLE_IDS_IN_ERROR = 5


def _copy_metadata_fields(
    scorer_sample: Dict[str, Any],
    metadata: Dict[str, Any],
    fields: Tuple[str, ...],
) -> None:
    for field in fields:
        if field in metadata:
            scorer_sample[field] = metadata[field]


def _token_verdict(token: str) -> Optional[bool]:
    """Map a normalized (stripped, lower-cased) token to True/False, or None if unrecognized."""
    if token in JUDGE_TRUE_TOKENS:
        return True
    if token in JUDGE_FALSE_TOKENS:
        return False
    return None


def _strip_code_fence(text: str) -> str:
    """Return the content of a ``` fenced block (optionally tagged, e.g. ```json), else the text itself."""
    match = JUDGE_CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def parse_judge_verdict(
    verdict: Optional[str],
    raw_judge_output: Optional[str],
) -> Optional[bool]:
    """Normalize a judge response to True (correct), False (incorrect) or None (unparsed).

    Precedence, first hit wins; every comparison is case-insensitive on stripped text:
      1. ``verdict``: 'correct'/'true'/'yes' -> True; 'incorrect'/'false'/'no' -> False.
         An empty or unrecognized verdict falls through to the raw output.
      2. ``raw_judge_output`` parsed as JSON (``` fences are stripped first): a dict whose
         'answer' is a bool or one of the tokens above, or a top-level JSON bool.
      3. An ``answer: true|false|yes|no`` line anywhere in the raw output.
      4. The whole raw output being one of the tokens above (a trailing period is ignored).

    Anything else (empty output, prose, rubric JSON without an 'answer' key, ...) returns
    None so callers can count it instead of silently treating it as incorrect.

    Args:
        verdict: Optional pre-parsed label supplied by the caller
        raw_judge_output: The judge model's text

    Returns:
        True, False, or None when no verdict could be read
    """
    result = _token_verdict((verdict or "").strip().lower())
    if result is not None:
        return result

    raw = _strip_code_fence((raw_judge_output or "").strip())
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, dict) and "answer" in parsed:
        answer = parsed["answer"]
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str):
            result = _token_verdict(answer.strip().lower())
            if result is not None:
                return result

    answer_line = JUDGE_ANSWER_LINE_RE.search(raw)
    if answer_line:
        return answer_line.group(1).lower() in JUDGE_TRUE_TOKENS

    return _token_verdict(raw.lower().rstrip("."))


def _summarize_judge_details(judge_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate judge-mode score_details; missing and unparsed rows count in the denominator."""
    judge_samples = len(judge_details)
    judge_correct = sum(details["score"] for details in judge_details)
    return {
        "judge_accuracy": judge_correct / judge_samples if judge_samples > 0 else 0.0,
        "judge_samples": judge_samples,
        "judge_missing_responses": sum(
            1 for details in judge_details if details["judge_missing_response"]
        ),
        "judge_unparsed_verdicts": sum(
            1 for details in judge_details
            if details["judge_verdict"] == JUDGE_VERDICT_UNPARSED
        ),
    }


def _preview_sample_ids(sample_ids: List[str]) -> str:
    preview = ", ".join(sample_ids[:MAX_SAMPLE_IDS_IN_ERROR])
    if len(sample_ids) > MAX_SAMPLE_IDS_IN_ERROR:
        preview += ", ..."
    return preview


class DatBenchEvaluator:
    """Main evaluation harness for DatBench."""

    def __init__(self, hf_dataset, capability: str):
        """Initialize evaluator with HF dataset.

        Args:
            hf_dataset: HuggingFace dataset (from load_dataset)
            capability: Capability name (e.g., "math", "document")
        """
        self.capability = capability
        self.hf_dataset = hf_dataset
        self.samples = [
            DatBenchSample.from_hf_row(row, capability)
            for row in hf_dataset
        ]
        self.samples_by_id = {s.id: s for s in self.samples}

    def get_inference_tasks(self) -> List[InferenceTask]:
        """Get list of inference tasks for the VLM."""
        return [InferenceTask.from_sample(s) for s in self.samples]

    def create_judge_tasks(
        self,
        vlm_responses: List[VLMResponse]
    ) -> List[JudgeRequest]:
        """Create judge tasks for samples that need judge-based evaluation."""
        vlm_by_id = {r.id: r for r in vlm_responses}
        judge_tasks = []

        for sample in self.samples:
            if sample.eval_mode == "judge":
                vlm_response = vlm_by_id.get(sample.id)
                if vlm_response:
                    judge_tasks.append(
                        JudgeRequest.from_sample_and_response(sample, vlm_response)
                    )

        return judge_tasks

    @staticmethod
    def _score_judge_sample(judge_response: Optional[JudgeResponse]) -> Dict[str, Any]:
        """Score a judge-mode sample from its judge response alone.

        Dataset scorers are never consulted for judge-mode samples: a missing response
        scores 0.0 with judge_verdict 'missing', an unreadable one scores 0.0 with
        judge_verdict 'unparsed'.
        """
        if judge_response is None:
            return {
                "score": 0.0,
                "eval_mode": JUDGE_EVAL_MODE,
                "match": False,
                "judge_verdict": JUDGE_VERDICT_MISSING,
                "judge_raw_output": "",
                "judge_missing_response": True,
            }

        match = parse_judge_verdict(judge_response.verdict, judge_response.raw_judge_output)
        if match is None:
            judge_verdict = JUDGE_VERDICT_UNPARSED
        elif match:
            judge_verdict = JUDGE_VERDICT_CORRECT
        else:
            judge_verdict = JUDGE_VERDICT_INCORRECT
        return {
            "score": 1.0 if match else 0.0,
            "eval_mode": JUDGE_EVAL_MODE,
            "match": bool(match),
            "judge_verdict": judge_verdict,
            "judge_raw_output": judge_response.raw_judge_output or "",
            "judge_missing_response": False,
        }

    def compute_metrics(
        self,
        vlm_responses: List[VLMResponse],
        judge_responses: Optional[List[JudgeResponse]] = None,
        strict_judge: bool = False,
    ) -> DatBenchReport:
        """Score VLM responses and build the report.

        Direct-mode samples are scored by their dataset scorer. Judge-mode samples are
        scored only from the JudgeResponse with the same id (see parse_judge_verdict):
        1.0 for a correct verdict, 0.0 otherwise. A judge-mode sample with no judge
        response, or one whose verdict cannot be read, scores 0.0 and is counted in the
        summary (judge_missing_responses / judge_unparsed_verdicts). Judge responses whose
        id does not belong to a judge-mode sample are ignored.

        Args:
            vlm_responses: List of VLM responses
            judge_responses: List of judge responses for the judge-mode samples
            strict_judge: Raise ValueError instead of scoring 0.0 when a judge-mode
                sample with a VLM response has no judge response or an unparsed verdict

        Returns:
            DatBenchReport with summary and per-sample scores

        Raises:
            ValueError: In strict_judge mode, for missing or unparsed judge verdicts
        """
        vlm_by_id = {r.id: r for r in vlm_responses}
        judge_by_id = {r.id: r for r in (judge_responses or [])}

        if strict_judge:
            missing_judge_ids = [
                sample.id for sample in self.samples
                if sample.eval_mode == JUDGE_EVAL_MODE
                and sample.id in vlm_by_id
                and sample.id not in judge_by_id
            ]
            if missing_judge_ids:
                raise ValueError(
                    f"Missing judge responses for {len(missing_judge_ids)} judge-mode "
                    f"sample(s): {_preview_sample_ids(missing_judge_ids)}"
                )

        # Score each sample using dataset-specific scorer
        sample_scores = []
        detailed_metrics_by_dataset = defaultdict(list)
        judge_details_by_dataset = defaultdict(list)
        unparsed_judge_ids = []

        for sample in self.samples:
            vlm_response = vlm_by_id.get(sample.id)
            if not vlm_response:
                continue

            dataset_name = sample.source_info.get("dataset", "")
            model_output = vlm_response.parsed_answer or vlm_response.raw_output

            # Get dataset-specific scoring module
            scoring_module = DATASET_SCORING_MODULES.get(dataset_name)

            if sample.eval_mode == JUDGE_EVAL_MODE:
                # Judge-mode samples are scored from the verdict only; some judge-only
                # datasets have no direct scorer at all
                score_details = self._score_judge_sample(judge_by_id.get(sample.id))
                judge_details_by_dataset[dataset_name].append(score_details)
                if score_details["judge_verdict"] == JUDGE_VERDICT_UNPARSED:
                    unparsed_judge_ids.append(sample.id)

            elif scoring_module and hasattr(scoring_module, 'score_sample'):
                # Convert sample to format expected by scorer
                scorer_sample = self._convert_sample_for_scorer(sample)
                score_details = scoring_module.score_sample(scorer_sample, model_output)

                # Store detailed metrics for aggregation
                detailed_metrics_by_dataset[dataset_name].append(score_details)
            else:
                # Fallback: basic exact match
                extracted = extract_final_answer(model_output)
                exact_match = extracted.strip().lower() == sample.answer.strip().lower()
                score_details = {'score': 1.0 if exact_match else 0.0}

            # Extract binary correctness
            is_correct = score_details.get('score', 0.0) >= 0.5

            sample_scores.append(SampleScore(
                id=sample.id,
                score=float(score_details.get('score', 0.0)),
                vlm_output=model_output,
                ground_truth=sample.answer,
                is_correct=is_correct,
                metadata={
                    "capability": sample.capability,
                    "eval_mode": sample.eval_mode,
                    "dataset": dataset_name,
                    "is_circular": sample.is_circular,
                    "discrimination": sample.eval_metrics.get("discrimination", 0.0),
                    "score_details": score_details,  # Include full details
                }
            ))

        if strict_judge and unparsed_judge_ids:
            raise ValueError(
                f"Could not read the judge verdict for {len(unparsed_judge_ids)} judge-mode "
                f"sample(s): {_preview_sample_ids(unparsed_judge_ids)}"
            )

        judge_details = [
            details
            for details_list in judge_details_by_dataset.values()
            for details in details_list
        ]
        judge_summary = _summarize_judge_details(judge_details)
        if judge_summary["judge_samples"] > 0 and (
            judge_summary["judge_missing_responses"] == judge_summary["judge_samples"]
        ):
            warnings.warn(
                f"{judge_summary['judge_samples']} judge-mode sample(s) were scored 0.0 "
                "because no judge response matched them; run create_judge_tasks() and pass "
                "judge_responses to compute_metrics()",
                UserWarning,
            )

        # Compute aggregated metrics per dataset
        aggregated_dataset_metrics = {}
        for dataset_name, score_details_list in detailed_metrics_by_dataset.items():
            scoring_module = DATASET_SCORING_MODULES.get(dataset_name)
            if scoring_module and hasattr(scoring_module, 'compute_metrics'):
                # Create predictions in format expected by compute_metrics
                predictions = [{'score_details': sd} for sd in score_details_list]
                dataset_metrics = scoring_module.compute_metrics(predictions)
                aggregated_dataset_metrics[dataset_name] = dataset_metrics

        # Judge-mode rows never reach a dataset scorer's compute_metrics; the judge_*
        # prefix keeps these keys apart from scorer keys when a dataset has both modes
        for dataset_name, judge_details_list in judge_details_by_dataset.items():
            aggregated_dataset_metrics.setdefault(dataset_name, {}).update(
                _summarize_judge_details(judge_details_list)
            )

        # Compute overall summary
        total_correct = sum(s.score for s in sample_scores)
        total_samples = len(sample_scores)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        direct_scores = [
            s.score for s in sample_scores if s.metadata["eval_mode"] != JUDGE_EVAL_MODE
        ]
        direct_samples = len(direct_scores)
        direct_accuracy = sum(direct_scores) / direct_samples if direct_samples > 0 else 0.0

        # Per-capability breakdown
        by_capability = defaultdict(lambda: {"correct": 0, "total": 0})
        for score in sample_scores:
            cap = score.metadata["capability"]
            by_capability[cap]["total"] += 1
            by_capability[cap]["correct"] += score.score

        capability_accuracy = {
            cap: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for cap, stats in by_capability.items()
        }

        summary = {
            "overall_accuracy": accuracy,
            "total_samples": total_samples,
            "total_correct": int(total_correct),
            "capability_accuracy": capability_accuracy,
            "capability_counts": {cap: stats["total"] for cap, stats in by_capability.items()},
            "dataset_metrics": aggregated_dataset_metrics,  # Add detailed per-dataset metrics
            "direct_samples": direct_samples,
            "direct_accuracy": direct_accuracy,
            **judge_summary,
        }

        return DatBenchReport(
            summary=summary,
            results=sample_scores
        )


    def _convert_sample_for_scorer(self, sample: DatBenchSample) -> Dict[str, Any]:
        """Convert DatBenchSample to format expected by scoring modules.

        Extracts dataset-specific metadata and prepares sample for scoring.

        Args:
            sample: DatBenchSample instance

        Returns:
            Dict in format expected by dataset.score_sample()

        Raises:
            ValueError: If required metadata field is missing
        """
        dataset_name = sample.source_info.get('dataset', '')

        # Parse metadata from JSON string
        metadata_raw = sample.metadata if hasattr(sample, 'metadata') else '{}'
        if isinstance(metadata_raw, str):
            metadata = json.loads(metadata_raw)
        else:
            metadata = metadata_raw or {}

        # Base scorer sample with standard fields
        scorer_sample = {
            "question_id": sample.id,
            "question": sample.question,
            "answer": sample.answer,
            "all_answers": sample.all_answers,
        }
        _copy_metadata_fields(scorer_sample, metadata, COMMON_MCQ_METADATA_FIELDS)

        # Dataset-specific metadata extraction

        if dataset_name == 'tallyqa':
            if 'is_simple' not in metadata:
                raise ValueError(f"TallyQA sample {sample.id} missing required metadata field: is_simple")
            scorer_sample["is_simple"] = metadata["is_simple"]

        elif dataset_name == 'vqa-v2':
            if 'answer_type' not in metadata or 'question_type' not in metadata:
                raise ValueError(f"VQAv2 sample {sample.id} missing required metadata: answer_type, question_type")
            scorer_sample["answer_type"] = metadata["answer_type"]
            scorer_sample["question_type"] = metadata["question_type"]

        elif dataset_name.startswith('refcoco'):
            if 'bbox' not in metadata:
                raise ValueError(f"RefCOCO sample {sample.id} missing required metadata field: bbox")
            scorer_sample["bbox"] = metadata["bbox"]

        elif dataset_name == 'pixmo_points_eval':
            required = ['points_target', 'points_mask', 'has_target']
            missing = [f for f in required if f not in metadata]
            if missing:
                raise ValueError(f"Pixmo sample {sample.id} missing required metadata: {missing}")
            scorer_sample["points"] = metadata["points_target"]
            scorer_sample["masks"] = metadata["points_mask"]
            scorer_sample["has_target"] = metadata["has_target"]

        elif dataset_name == 'mathvision':
            if 'options' in metadata:
                scorer_sample["options"] = metadata["options"]
            if 'options_struct' in metadata:
                scorer_sample["options_struct"] = metadata["options_struct"]
            if 'correct_choice_letter' in metadata:
                scorer_sample["correct_choice_letter"] = metadata["correct_choice_letter"]

        elif dataset_name == 'logicvista':
            scorer_sample["question"] = sample.question
            if 'options' in metadata:
                scorer_sample["options"] = metadata["options"]

        elif dataset_name in ['cc-ocr-doc_parsing', 'cc-ocr-kie', 'cc-ocr-multi_scene_ocr']:
            if 'l2_category' in metadata:
                scorer_sample["l2_category"] = metadata["l2_category"]

        elif dataset_name == 'ocrbench-v2':
            if 'type' in metadata:
                scorer_sample["type"] = metadata["type"]
            if 'eval' in metadata:
                scorer_sample["eval"] = metadata["eval"]
            if 'bbox' in metadata:
                scorer_sample["bbox"] = metadata["bbox"]
            if 'bbox_list' in metadata:
                scorer_sample["bbox_list"] = metadata["bbox_list"]
            if 'content' in metadata:
                scorer_sample["content"] = metadata["content"]
            scorer_sample["answers"] = sample.all_answers
            scorer_sample["question"] = sample.question

        elif dataset_name == 'chartqapro':
            if 'question_type' in metadata:
                scorer_sample["question_type"] = metadata["question_type"]
            if 'year_flags' in metadata:
                scorer_sample["year_flags"] = metadata["year_flags"]
            if 'is_multi_turn' in metadata:
                scorer_sample["is_multi_turn"] = metadata["is_multi_turn"]
            metadata_answers = metadata.get("all_answers")
            if isinstance(metadata_answers, list) and metadata_answers:
                canonical_answers = metadata_answers
            elif sample.all_answers:
                canonical_answers = sample.all_answers
            else:
                canonical_answers = [sample.answer]
            scorer_sample["all_answers"] = canonical_answers

        elif dataset_name in ['charxiv_descriptive', 'charxiv_reasoning']:
            if 'question_type' in metadata:
                scorer_sample["question_type"] = metadata["question_type"]

        return scorer_sample
