from datbench import DatBenchEvaluator, JudgeResponse, VLMResponse
from scripts.convert_general_vqav2_to_judge import convert_row, is_vqav2_general_row


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


def test_release_converter_targets_only_general_vqav2_rows():
    target = _row("vqa", "vqa-v2", "cat", ["cat"])
    other = _row("mmbench", "mmbench", "A", ["A"])

    assert is_vqav2_general_row(target)
    assert not is_vqav2_general_row(other)

    converted = convert_row(target)
    untouched = convert_row(other)

    assert converted["eval_mode"] == "judge"
    assert "semantic correctness" in converted["judge_prompt"]
    assert untouched["eval_mode"] == "direct"
    assert untouched["judge_prompt"] == ""


def test_judge_mode_vqav2_rows_create_judge_tasks():
    row = _row(
        "vqa",
        "vqa-v2",
        "cat",
        ["cat", "cat", "kitten"],
        eval_mode="judge",
        judge_prompt="Judge it.",
    )
    evaluator = DatBenchEvaluator([row], "general")

    tasks = evaluator.create_judge_tasks([VLMResponse(id="vqa", raw_output="kitten")])

    assert len(tasks) == 1
    assert tasks[0].id == "vqa"
    assert tasks[0].vlm_response == "kitten"
    assert tasks[0].ground_truth == "cat"


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
