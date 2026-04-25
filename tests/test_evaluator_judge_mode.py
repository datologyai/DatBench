from datbench import DatBenchEvaluator, JudgeResponse, VLMResponse
from datbench.judge_policies.vqav2 import (
    VQA_V2_JUDGE_FINAL_ANSWER_POLICY,
    vqa_v2_judge_prompt_for_sample,
    with_vqa_v2_final_answer_policy,
)
from datbench.schema import JudgeRequest


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


def test_vqav2_policy_helper_adds_final_answer_policy():
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

    assert VQA_V2_JUDGE_FINAL_ANSWER_POLICY in judge_prompt
    assert "\\boxed{...}" in judge_prompt


def test_judge_request_uses_row_prompt_without_source_specific_mutation():
    row = _row(
        "judge-vqa",
        "vqa-v2",
        "none",
        ["none", "nothing"],
        eval_mode="judge",
        judge_prompt=with_vqa_v2_final_answer_policy(
            "Judge semantic VQA correctness."
        ),
    )
    sample = DatBenchEvaluator([row], "general").samples_by_id["judge-vqa"]

    request = JudgeRequest.from_sample_and_response(
        sample,
        VLMResponse(id="judge-vqa", raw_output="\\boxed{None}"),
    )

    assert VQA_V2_JUDGE_FINAL_ANSWER_POLICY in request.judge_prompt
    assert request.judge_prompt == sample.judge_prompt


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

    assert VQA_V2_JUDGE_FINAL_ANSWER_POLICY not in judge_prompt
    assert judge_prompt == "Judge chart correctness."
