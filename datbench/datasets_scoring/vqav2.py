"""VQAv2 scoring.

Uses VQAv2 soft accuracy: for each answer position i, count how many
of the other 9 annotator answers match the prediction, divide by 3,
take min(1, _), then average across all 10 positions.
"""

import re
from typing import Any, Dict, List
from .evaluation_utils.erma_utils import extract_final_answer
from .evaluation_utils.vqav2_eval import (
    process_punctuation,
    process_digits_articles_contractions,
)


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a VQAv2 sample using soft accuracy.

    Args:
        sample: Sample data with 'all_answers' (list of 10 annotator answers)
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Extract concise answer before scoring
    pred_answer = extract_final_answer(model_output)

    # Get all answers (should be 10 annotator answers for VQAv2)
    all_answers = sample.get("all_answers", sample.get("answers", []))

    # Compute VQAv2 soft accuracy
    score = _compute_vqav2_soft_accuracy(pred_answer, all_answers)

    # Get metadata for aggregation
    answer_type = sample.get("answer_type", "unknown")
    question_type = sample.get("question_type", "unknown")

    return {
        "score": score,
        "ground_truth_answers": all_answers,
        "model_output": model_output,
        "pred_answer": pred_answer,
        "answer_type": answer_type,
        "question_type": question_type,
    }


def _compute_vqav2_soft_accuracy(prediction: str, all_answers: List[str]) -> float:
    """Compute VQAv2 soft accuracy following official evaluation.

    For each of the 10 annotator answers (position i):
    - Count how many of the OTHER 9 answers match the prediction
    - Accuracy_i = min(1, matches/3)
    - Final score = average of all 10 Accuracy_i values

    Args:
        prediction: Model's predicted answer
        all_answers: List of 10 annotator answers

    Returns:
        Soft accuracy score [0, 1]
    """
    if not all_answers:
        return 0.0

    # Normalize prediction using VQAv2 answer processing
    pred_norm = _process_answer(prediction)

    # Normalize all ground truth answers
    gts_norm = [_process_answer(ans) for ans in all_answers]

    # Compute accuracy for each annotator position
    accuracies = []
    n = len(gts_norm)

    for i in range(n):
        # Get other answers (excluding position i)
        other_answers = [gts_norm[j] for j in range(n) if j != i]

        # Count matches with prediction
        matches = sum(1 for ans in other_answers if ans == pred_norm)

        # Accuracy is min(1, matches/3)
        acc_i = min(1.0, float(matches) / 3.0)
        accuracies.append(acc_i)

    # Return average accuracy across all positions
    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def _process_answer(answer: str) -> str:
    """Process answer using VQAv2 normalization rules.

    Follows official VQAv2 evaluation:
    1. Convert to lowercase, strip whitespace
    2. Process punctuation
    3. Process articles, digits, contractions
    """
    if not answer:
        return ""

    # Convert to lowercase and strip
    text = answer.lower().strip()

    # Replace newlines and tabs with space
    text = text.replace("\n", " ").replace("\t", " ").strip()

    # Process punctuation
    text = process_punctuation(text)

    # Process digits, articles, and contractions
    text = process_digits_articles_contractions(text)

    return text


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute VQAv2 metrics with breakdowns by answer_type and question_type.

    Args:
        results: List of result dictionaries with scores

    Returns:
        Dictionary with overall accuracy and breakdowns
    """
    scores_by_answer_type = {}
    scores_by_question_type = {}
    all_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        score = score_details.get("score", 0.0)
        answer_type = score_details.get("answer_type", "unknown")
        question_type = score_details.get("question_type", "unknown")

        all_scores.append(score)

        # Group by answer type
        if answer_type not in scores_by_answer_type:
            scores_by_answer_type[answer_type] = []
        scores_by_answer_type[answer_type].append(score)

        # Group by question type
        if question_type not in scores_by_question_type:
            scores_by_question_type[question_type] = []
        scores_by_question_type[question_type].append(score)

    # Compute overall accuracy
    overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Compute accuracy by answer type
    accuracy_by_answer_type = {
        ans_type: sum(scores) / len(scores)
        for ans_type, scores in scores_by_answer_type.items()
        if scores
    }

    # Compute accuracy by question type
    accuracy_by_question_type = {
        q_type: sum(scores) / len(scores)
        for q_type, scores in scores_by_question_type.items()
        if scores
    }

    return {
        "accuracy": overall_accuracy,
        "accuracy-by-answer-type": accuracy_by_answer_type,
        "accuracy-by-question-type": accuracy_by_question_type,
    }
