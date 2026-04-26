import pytest
import pyarrow as pa

from datbench import DatBenchEvaluator, JudgeResponse, VLMResponse
from datbench.judge_policies.vqav2 import (
    LEGACY_VQA_V2_POLICY_MARKERS,
    VQA_V2_SEMANTIC_JUDGE_POLICY,
    vqa_v2_judge_prompt_for_sample,
    with_vqa_v2_semantic_judge_policy,
)
from datbench.schema import JudgeRequest
from scripts.release.apply_vqav2_judge_prompt_policy import rewrite_table


def _row(
    sample_id,
    dataset,
    answer,
    all_answers,
    eval_mode="direct",
    judge_prompt="",
):
    return {
        "id": sample_id,
        "image": None,
        "question": "What is shown?",
        "prompt_format": {"prefix": "", "suffix": ""},
        "answer": answer,
        "all_answers": all_answers,
        "eval_mode": eval_mode,
        "judge_prompt": judge_prompt,
        "is_circular": False,
        "metadata": '{"answer_type": "other", "question_type": "what"}',
        "source_info": {"dataset": dataset, "original_idx": sample_id},
        "eval_metrics": {},
    }


def test_mixed_direct_and_judge_rows_aggregate_correctly():
    direct = _row(
        "direct-vqa",
        "vqa-v2",
        "cat",
        ["cat", "cat", "cat", "cat", "cat", "cat", "cat", "cat", "cat", "cat"],
    )
    judge = _row(
        "judge-vqa",
        "vqa-v2",
        "dog",
        ["dog", "dog", "puppy"],
        eval_mode="judge",
        judge_prompt="Judge it.",
    )
    evaluator = DatBenchEvaluator([direct, judge], "general")

    report = evaluator.compute_metrics(
        [
            VLMResponse(id="direct-vqa", raw_output="cat"),
            VLMResponse(id="judge-vqa", raw_output="puppy"),
        ],
        judge_responses=[
            JudgeResponse(
                id="judge-vqa",
                raw_judge_output='{"answer": true}',
                verdict="correct",
            )
        ],
    )

    assert report.summary["total_samples"] == 2
    assert report.summary["overall_accuracy"] == 1.0
    assert report.summary["dataset_metrics"]["vqa-v2"]["accuracy"] == 1.0
    assert {result.id: result.score for result in report.results} == {
        "direct-vqa": 1.0,
        "judge-vqa": 1.0,
    }


def test_judge_scoring_accepts_raw_boolean_outputs():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "dog",
        ["dog", "dog", "puppy"],
        eval_mode="judge",
        judge_prompt="Judge it.",
    )
    evaluator = DatBenchEvaluator([row], "general")

    report = evaluator.compute_metrics(
        [VLMResponse(id="judge-vqa", raw_output="puppy")],
        judge_responses=[JudgeResponse(id="judge-vqa", raw_judge_output="true")],
    )

    assert report.summary["overall_accuracy"] == 1.0
    assert report.results[0].metadata["score_details"]["judge_verdict"] is None


def test_malformed_raw_judge_output_scores_incorrect():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "dog",
        ["dog", "dog", "puppy"],
        eval_mode="judge",
        judge_prompt="Judge it.",
    )
    evaluator = DatBenchEvaluator([row], "general")

    report = evaluator.compute_metrics(
        [VLMResponse(id="judge-vqa", raw_output="puppy")],
        judge_responses=[
            JudgeResponse(id="judge-vqa", raw_judge_output="The answer is correct.")
        ],
    )

    assert report.summary["overall_accuracy"] == 0.0
    assert report.results[0].score == 0.0


def test_missing_judge_response_fails_loudly():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "dog",
        ["dog", "dog", "puppy"],
        eval_mode="judge",
        judge_prompt="Judge it.",
    )
    evaluator = DatBenchEvaluator([row], "general")

    with pytest.raises(ValueError, match="Missing judge response"):
        evaluator.compute_metrics([VLMResponse(id="judge-vqa", raw_output="puppy")])


def test_vqav2_direct_scorer_remains_available_for_unconverted_rows():
    row = _row(
        "direct-vqa",
        "vqa-v2",
        "red",
        ["red", "red", "red", "red", "red", "red", "red", "red", "red", "red"],
    )
    evaluator = DatBenchEvaluator([row], "general")

    report = evaluator.compute_metrics([VLMResponse(id="direct-vqa", raw_output="red")])

    assert report.summary["overall_accuracy"] == 1.0
    assert report.results[0].metadata["eval_mode"] == "direct"
    assert report.results[0].metadata["score_details"]["answer_type"] == "other"


def test_non_vqav2_judge_scoring_does_not_require_dataset_scorer_metadata():
    row = _row(
        "judge-refcoco",
        "refcoco_testA",
        "box",
        ["box"],
        eval_mode="judge",
        judge_prompt="Judge grounding correctness.",
    )
    evaluator = DatBenchEvaluator([row], "grounding")

    report = evaluator.compute_metrics(
        [VLMResponse(id="judge-refcoco", raw_output="box")],
        judge_responses=[JudgeResponse(id="judge-refcoco", raw_judge_output="true")],
    )

    assert report.summary["overall_accuracy"] == 1.0
    assert report.results[0].metadata["score_details"] == {
        "score": 1.0,
        "judge_verdict": None,
    }


def test_vqav2_policy_helper_adds_semantic_judge_policy():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "none",
        ["none", "nothing"],
        eval_mode="judge",
        judge_prompt="Judge semantic VQA correctness.",
    )
    sample = DatBenchEvaluator([row], "general").samples_by_id["judge-vqa"]

    judge_prompt = vqa_v2_judge_prompt_for_sample(sample)

    assert VQA_V2_SEMANTIC_JUDGE_POLICY in judge_prompt
    assert "Accept compatible refinements and hyponyms" in judge_prompt


def test_judge_request_uses_row_prompt_without_source_specific_mutation():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "none",
        ["none", "nothing"],
        eval_mode="judge",
        judge_prompt=with_vqa_v2_semantic_judge_policy(
            "Judge semantic VQA correctness."
        ),
    )
    sample = DatBenchEvaluator([row], "general").samples_by_id["judge-vqa"]

    request = JudgeRequest.from_sample_and_response(
        sample,
        VLMResponse(id="judge-vqa", raw_output="\\boxed{None}"),
    )

    assert VQA_V2_SEMANTIC_JUDGE_POLICY in request.judge_prompt
    assert request.judge_prompt == sample.judge_prompt


def test_vqav2_policy_helper_replaces_legacy_policy():
    legacy_prompt = (
        "Judge semantic VQA correctness.\n\n"
        f"{LEGACY_VQA_V2_POLICY_MARKERS[0]}:\n"
        "- Keep the semantic comparison strict for answer content."
    )

    judge_prompt = with_vqa_v2_semantic_judge_policy(legacy_prompt)

    assert "Judge semantic VQA correctness." in judge_prompt
    assert LEGACY_VQA_V2_POLICY_MARKERS[0] not in judge_prompt
    assert "Accept compatible refinements and hyponyms" in judge_prompt


def test_non_vqav2_policy_helper_does_not_add_vqa_policy():
    row = _row(
        "judge-chart",
        "chartqa",
        "5",
        ["5"],
        eval_mode="judge",
        judge_prompt="Judge chart correctness.",
    )
    sample = DatBenchEvaluator([row], "general").samples_by_id["judge-chart"]

    judge_prompt = vqa_v2_judge_prompt_for_sample(sample)

    assert VQA_V2_SEMANTIC_JUDGE_POLICY not in judge_prompt
    assert judge_prompt == "Judge chart correctness."


def test_release_rewrite_rejects_vqa_policy_on_non_target_rows():
    table = pa.table(
        {
            "id": ["chart-row"],
            "source_info": [{"dataset": "chartqa"}],
            "eval_mode": ["judge"],
            "judge_prompt": [
                f"Judge chart correctness.\n\n{LEGACY_VQA_V2_POLICY_MARKERS[0]}"
            ],
        }
    )

    with pytest.raises(ValueError, match="non-target row"):
        rewrite_table(table, path="memory.parquet")


def test_release_rewrite_updates_only_target_vqav2_rows():
    table = pa.table(
        {
            "id": ["vqa-row", "chart-row"],
            "source_info": [
                {"dataset": "vqa-v2"},
                {"dataset": "chartqa"},
            ],
            "eval_mode": ["direct", "judge"],
            "judge_prompt": ["Judge VQA correctness.", "Judge chart correctness."],
        }
    )

    rewritten, counts = rewrite_table(table, path="memory.parquet")
    eval_modes = rewritten["eval_mode"].to_pylist()
    prompts = rewritten["judge_prompt"].to_pylist()

    assert counts["target_rows"] == 1
    assert counts["converted_eval_mode_rows"] == 1
    assert counts["target_policy_rows"] == 1
    assert counts["non_target_policy_rows"] == 0
    assert eval_modes == ["judge", "judge"]
    assert VQA_V2_SEMANTIC_JUDGE_POLICY in prompts[0]
    assert prompts[1] == "Judge chart correctness."
