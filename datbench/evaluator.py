"""DatBench evaluation harness."""

import json
from typing import List, Dict, Optional, Any
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

    def compute_metrics(
        self,
        vlm_responses: List[VLMResponse],
        judge_responses: Optional[List[JudgeResponse]] = None,
    ) -> DatBenchReport:
        """
        Args:
            vlm_responses: List of VLM responses
            judge_responses: List of judge responses (optional, for judge-based eval)

        Returns:
            DatBenchReport with summary and per-sample scores
        """
        vlm_by_id = {r.id: r for r in vlm_responses}
        judge_by_id = {r.id: r for r in (judge_responses or [])}

        # Score each sample using dataset-specific scorer
        sample_scores = []
        detailed_metrics_by_dataset = defaultdict(list)

        for sample in self.samples:
            vlm_response = vlm_by_id.get(sample.id)
            if not vlm_response:
                continue

            dataset_name = sample.source_info.get("dataset", "")
            model_output = vlm_response.parsed_answer or vlm_response.raw_output

            # Get dataset-specific scoring module
            scoring_module = DATASET_SCORING_MODULES.get(dataset_name)

            if scoring_module and hasattr(scoring_module, 'score_sample'):
                # Convert sample to format expected by scorer
                scorer_sample = self._convert_sample_for_scorer(sample)
                score_details = scoring_module.score_sample(scorer_sample, model_output)

                # Extract binary correctness
                is_correct = score_details.get('score', 0.0) >= 0.5

                # Store detailed metrics for aggregation
                detailed_metrics_by_dataset[dataset_name].append(score_details)
            else:
                # Fallback: basic exact match
                extracted = extract_final_answer(model_output)
                is_correct = extracted.strip().lower() == sample.answer.strip().lower()
                score_details = {'score': 1.0 if is_correct else 0.0}

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

        # Compute aggregated metrics per dataset
        aggregated_dataset_metrics = {}
        for dataset_name, score_details_list in detailed_metrics_by_dataset.items():
            scoring_module = DATASET_SCORING_MODULES.get(dataset_name)
            if scoring_module and hasattr(scoring_module, 'compute_metrics'):
                # Create predictions in format expected by compute_metrics
                predictions = [{'score_details': sd} for sd in score_details_list]
                dataset_metrics = scoring_module.compute_metrics(predictions)
                aggregated_dataset_metrics[dataset_name] = dataset_metrics

        # Compute overall summary
        total_correct = sum(s.score for s in sample_scores)
        total_samples = len(sample_scores)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

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

        elif dataset_name == 'mmbench':
            if 'options' in metadata:
                scorer_sample["options"] = metadata["options"]
            if 'index2ans' in metadata:
                scorer_sample["index2ans"] = metadata["index2ans"]
            if 'all_choices' in metadata:
                scorer_sample["all_choices"] = metadata["all_choices"]

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
            scorer_sample["Answer"] = canonical_answers
            scorer_sample["ground_truth_answer"] = canonical_answers

        elif dataset_name in ['charxiv_descriptive', 'charxiv_reasoning']:
            if 'question_type' in metadata:
                scorer_sample["question_type"] = metadata["question_type"]

        return scorer_sample
