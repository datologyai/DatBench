# DatBench Evaluation Library

Official evaluation harness for DatBench, a high-fidelity vision-language benchmark with exact scoring implementations.


## Quick Start

```python
from datasets import load_dataset
from datbench import DatBenchEvaluator, VLMResponse, JudgeResponse

# Load dataset
capability = "math"  # Options: chart, counting, document, general, grounding, math, scene, spatial, table
dataset = load_dataset("DatologyAI/DatBench", capability, split="test")

# Initialize evaluator
evaluator = DatBenchEvaluator(dataset, capability)

# Get inference tasks
tasks = evaluator.get_inference_tasks()

# Run your VLM
def run_my_vlm(task):
    # task.image is a PIL.Image.Image
    # task.question is the formatted prompt
    # Return your model's output string
    return model_output

# Create VLM responses
vlm_responses = [
    VLMResponse(id=task.id, raw_output=run_my_vlm(task))
    for task in tasks
]

# Run your judge LLM on the judge-mode samples (see "Judge-mode samples" below).
# chart, general, math and table contain judge-mode samples; without judge
# responses those samples score 0.
judge_tasks = evaluator.create_judge_tasks(vlm_responses)

def run_my_judge(task):
    # task.judge_prompt is the rubric; task.vlm_response and task.ground_truth are
    # the texts to compare. The sample's question and alternative references are
    # available via evaluator.samples_by_id[task.id].question / .all_answers.
    # Return the judge's text, e.g. '{"answer": true}' or 'answer: false'
    return judge_output

judge_responses = [
    JudgeResponse(id=task.id, raw_judge_output=run_my_judge(task))
    for task in judge_tasks
]

# Compute metrics
report = evaluator.compute_metrics(vlm_responses, judge_responses=judge_responses)

# View results
print(f"Accuracy: {report.summary['overall_accuracy']:.2%}")
print(f"Direct: {report.summary['direct_accuracy']:.2%} ({report.summary['direct_samples']} samples)")
print(f"Judge: {report.summary['judge_accuracy']:.2%} ({report.summary['judge_samples']} samples)")
print(f"Per-dataset: {report.summary['dataset_metrics']}")

# Save results
report.save("results.json")
```

## Judge-mode samples

Every sample has an `eval_mode`:

- `direct`: the sample is scored by the scoring module of its source dataset (exact match, ANLS, relaxed numeric match, IoU, ...).
- `judge`: the sample is scored **only** from the verdict of a judge LLM. The dataset scoring modules are never consulted for these samples.

`create_judge_tasks(vlm_responses)` returns one `JudgeRequest` per judge-mode sample that has a VLM response. Run your judge on each request and collect a `JudgeResponse(id=task.id, raw_judge_output=...)`. If you parse the judge output yourself, also pass `verdict="correct"` or `verdict="incorrect"`; it takes precedence over `raw_judge_output`, which stays required (pass `""` if you have no text to keep).

Each judge-mode sample is scored 1.0 or 0.0 by `parse_judge_verdict(verdict, raw_judge_output)`, which reads, in this order (first hit wins, case-insensitive):

1. the `verdict` field: `correct`/`true`/`yes` or `incorrect`/`false`/`no` (an unrecognized value falls through to the raw output);
2. the raw output as JSON (optionally inside a ``` code fence): `{"answer": true|false}`, `{"answer": "yes"|"no"|"correct"|"incorrect"}`, or a bare JSON boolean;
3. an `answer: true|false|yes|no` line anywhere in the raw output (so a two-line `Reason: ... / answer: true` reply works);
4. the whole raw output being one of the tokens `true`/`false`/`yes`/`no`/`correct`/`incorrect`.

Anything else (empty output, prose, rubric JSON without an `answer` key) is recorded as `unparsed` and scores 0. A judge-mode sample with a VLM response but no judge response scores 0, is recorded as `missing`, and still counts in `total_samples` and in the `judge_accuracy` denominator; `compute_metrics` emits a `UserWarning` when no judge response matched any judge-mode sample. Pass `strict_judge=True` to raise a `ValueError` instead for missing or unparsed verdicts.

Notes on the judge prompts:

- Prompts that end with a two-line `Reason: ... / answer: true or false` instruction are rewritten by `create_judge_tasks` to request a single boolean token, so a small `max_new_tokens` cannot truncate the verdict.
- Some prompts contain `{question}`, `{correct_answer}` and `{model_answer}` placeholders. Fill them yourself, or supply the question, the reference answer(s) and the model answer alongside the prompt when calling your judge.

## Dataset Information

### Available Datasets

DatBench provides two versions:

- **DatologyAI/DatBench**: High-fidelity subset (~5K samples per capability, ~45K total)
- **DatologyAI/DatBench-Full**: Complete dataset (~205K samples total)

### Capabilities

Nine evaluation capabilities covering diverse vision-language tasks:

- **chart**: Chart understanding, infographic QA
- **counting**: Object counting tasks
- **document**: OCR, document parsing, KIE
- **general**: General VQA, reasoning
- **grounding**: Referring expression grounding, point localization
- **math**: Mathematical reasoning, geometry
- **scene**: Scene text recognition, multi-scene OCR
- **spatial**: Spatial reasoning, real-world QA
- **table**: Table understanding, diagram QA

### Sample Structure

Each sample contains:

```python
{
    "id": "db_math_000123",           # Unique identifier
    "image": PIL.Image,               # Image (loaded automatically by HF)
    "question": str,                  # Formatted prompt ready for inference
    "answer": str,                    # Ground truth answer (the reference to give the judge)
    "all_answers": List[str],         # Alternative valid answers (also references for the judge)
    "eval_mode": "direct",            # "direct" or "judge"
    "judge_prompt": str,              # Rubric for the judge LLM (used by judge-mode samples)
    "is_circular": bool,              # Circular evaluation variant
    "metadata": str,                  # JSON string with dataset-specific metadata
    "source_info": {
        "dataset": str,               # Source dataset name
        "original_idx": str           # Original sample ID
    },
    "eval_metrics": Dict[str, Any]    # Per-sample statistics (e.g. discrimination)
}
```

## API Reference

### DatBenchEvaluator

Main evaluation class.

**Methods:**
- `__init__(hf_dataset, capability)` - Initialize with HF dataset
- `get_inference_tasks()` - Get list of InferenceTask objects
- `create_judge_tasks(vlm_responses)` - Create one JudgeRequest per judge-mode sample with a VLM response
- `compute_metrics(vlm_responses, judge_responses=None, strict_judge=False)` - Score and generate report. Direct-mode samples use their dataset scorer; judge-mode samples use the matching JudgeResponse. With `strict_judge=True` a missing or unparsed judge verdict raises `ValueError` instead of scoring 0.

**Attributes:**
- `samples` / `samples_by_id` - The loaded `DatBenchSample` objects (question, answer, all_answers, judge_prompt, ...)

### parse_judge_verdict

`parse_judge_verdict(verdict, raw_judge_output)` returns `True`, `False` or `None` (unparsed) using the precedence described under "Judge-mode samples". Use it to check that your judge's output format is understood before running a full evaluation.

### Dataclasses

**InferenceTask**: Input for your VLM
- `id`: Sample identifier
- `image`: PIL.Image.Image
- `question`: Formatted prompt string
- `eval_mode`: "direct" or "judge"

**VLMResponse**: Your VLM output
- `id`: Sample identifier
- `raw_output`: Full model response
- `parsed_answer`: Optional pre-extracted answer

**JudgeRequest**: Input for your judge LLM (judge-mode samples only)
- `id`: Sample identifier
- `vlm_response`: The model answer to judge (`parsed_answer` if set, else `raw_output`)
- `ground_truth`: The reference answer
- `judge_prompt`: The rubric

**JudgeResponse**: Your judge output
- `id`: Sample identifier
- `raw_judge_output`: The judge's text (parsed when `verdict` is not set)
- `verdict`: Optional `"correct"` / `"incorrect"` if you parse the judge output yourself

**SampleScore**: Per-sample result (`report.results`)
- `id`, `score`, `is_correct`, `vlm_output`, `ground_truth`
- `metadata`: capability, eval_mode, dataset, is_circular, discrimination and `score_details`. For direct-mode samples `score_details` is the dataset scorer's output; for judge-mode samples it holds `score`, `eval_mode` (`"judge"`), `match`, `judge_verdict` (`"correct"`, `"incorrect"`, `"missing"` or `"unparsed"`), `judge_raw_output` and `judge_missing_response`.

**DatBenchReport**: Final results
- `summary`: Dict with
  - `overall_accuracy`, `total_samples`, `total_correct` over all scored samples (direct and judge)
  - `capability_accuracy`, `capability_counts`
  - `direct_samples`, `direct_accuracy` over direct-mode samples
  - `judge_samples`, `judge_accuracy` over judge-mode samples (missing and unparsed verdicts count as 0 in the numerator and 1 in the denominator), `judge_missing_responses`, `judge_unparsed_verdicts`
  - `dataset_metrics`: per source dataset, the scorer's own metrics for its direct-mode samples, plus `judge_accuracy`, `judge_samples`, `judge_missing_responses` and `judge_unparsed_verdicts` for its judge-mode samples
- `results`: List[SampleScore] with per-sample details
- `save(path)`: Save report to JSON file

## Running tests

```bash
python -m unittest discover -s tests
```

## Citation

```bibtex
@misc{datologyai2026datbenchdiscriminativefaithfulefficient,
      title={DatBench: Discriminative, Faithful, and Efficient VLM Evaluations}, 
      author={DatologyAI and : and Siddharth Joshi and Haoli Yin and Rishabh Adiga and Ricardo Monti and Aldo Carranza and Alex Fang and Alvin Deng and Amro Abbas and Brett Larsen and Cody Blakeney and Darren Teh and David Schwab and Fan Pan and Haakon Mongstad and Jack Urbanek and Jason Lee and Jason Telanoff and Josh Wills and Kaleigh Mentzer and Luke Merrick and Parth Doshi and Paul Burstein and Pratyush Maini and Scott Loftin and Spandan Das and Tony Jiang and Vineeth Dorna and Zhengping Wang and Bogdan Gaza and Ari Morcos and Matthew Leavitt},
      year={2026},
      eprint={2601.02316},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02316}, 
}
```
