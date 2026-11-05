# AGENTS.md: wiring DatBench scoring into an eval stack

This file is for engineers and coding agents adding DatBench to an existing VLM evaluation harness
(a vLLM / HF Transformers / API runner, lmms-eval, VLMEvalKit, an in-house loop, ...). It states the
contract between your harness and the `datbench` package precisely enough that two independent
integrations produce the same number. The README covers the API; this file covers how to use it
correctly.

## The contract in eight rules

1. **Pin both versions.** Load the dataset at a fixed revision and install the package at a fixed
   commit. Record both next to every score. See [Versions](#versions).
2. **Prompt with `task.question` from `get_inference_tasks()`, never with the raw `question` column.**
   The task prompt is `prompt_format.prefix + question + prompt_format.suffix`; the prefix and suffix
   carry the answer-format instructions, the multiple-choice options and the coordinate format.
3. **One user turn per row: the row's image plus `task.question`.** See [Generation](#3-generation).
4. **Strip `<think>...</think>` from model outputs before scoring.** Pass the remaining text as
   `VLMResponse.raw_output` and leave `parsed_answer` unset.
5. **Judge the judge-mode rows.** `chart`, `general`, `math` and `table` contain rows whose score
   comes only from an LLM judge verdict. Without judge responses those rows score 0. Use the
   [reference judge setup](#5-judge-step) so your numbers are comparable.
6. **Score with `compute_metrics(..., strict_judge=True)`.** A missing or unparsable verdict then
   raises instead of silently counting as wrong.
7. **The capability score is `report.summary["overall_accuracy"]`.** It is the mean per-row score
   over all rows (direct and judge). Some direct scores are fractional (ANLS, IoU, OCR metrics).
8. **Check the report before trusting it.** See the [checklist](#checklist).

## Versions

| What | Pin | Notes |
|---|---|---|
| `DatologyAI/DatBench` | `e0733bd399daf9230e8779407f20847c1d44ec65` | 9 configs, 43,512 rows |
| `DatologyAI/DatBench-Full` | `ab78bde437b6c791d3108ca450b68375e5e0b4b0` | 9 configs, 151,750 rows; a superset of DatBench |
| `datbench` package | a commit or release tag of this repo | `__version__` 1.0.1; record the commit, not just the version string |
| Python | 3.10 - 3.12 | 3.8/3.9 fail on syntax; 3.13 fails importing `latex2sympy2`'s antlr runtime |
| `nltk` | `>=3.9.1,<3.10.3` | 3.10.3+ raises on long strings in `edit_distance` and aborts the whole `document` capability |

The dataset `main` branch moves when fixes are published, so a run without `revision=` is not
reproducible. Every row in DatBench also exists, identically, in DatBench-Full.

## 1. Install

```bash
pip install "datbench @ git+https://github.com/datologyai/DatBench.git@<commit-or-tag>" "nltk>=3.9.1,<3.10.3"
python -m nltk.downloader wordnet
```

- `wordnet` is needed for METEOR in OCRBench v2 full-page OCR (`document`). If it cannot be loaded,
  METEOR silently becomes 0 and `document` scores drop without an error. Some nltk versions also
  refuse to read a data directory that is group- or world-writable. Check with
  `python -c "from nltk.corpus import wordnet; wordnet.synsets('test')"`.
- `latex2sympy2` pins `antlr4-python3-runtime==4.7.2`, which conflicts with `omegaconf<2.4` /
  `hydra-core<1.4`. If your harness needs those, run scoring in its own environment or process: the
  scoring step only needs your saved outputs.
- The unit tests are not installed with the package; run them from a clone:
  `python -m unittest discover -s tests`.

## 2. Load the data

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

REPO = "DatologyAI/DatBench"
REVISION = "e0733bd399daf9230e8779407f20847c1d44ec65"
CAPABILITIES = ["chart", "counting", "document", "general", "grounding", "math", "scene", "spatial", "table"]

def load_capability(capability: str):
    root = snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=[f"{capability}/*"])
    return load_dataset("parquet", data_files=f"{root}/{capability}/*.parquet", split="train")
```

- `load_dataset(REPO, capability, split="test", revision=REVISION)` also works online. Loading the
  parquet files from a revision-pinned snapshot is safer on offline clusters: with
  `HF_HUB_OFFLINE=1`, `datasets` can ignore `revision=` and return whichever revision it cached
  last.
- `allow_patterns` downloads one configuration only. Sizes vary a lot (`scene` is about 18 GB of
  parquet, `math` about 0.3 GB).
- Each row has exactly one RGB image.

## 3. Generation

```python
from datbench import DatBenchEvaluator

evaluator = DatBenchEvaluator(dataset, capability)   # capability is a label; scorers are chosen per row
tasks = evaluator.get_inference_tasks()              # InferenceTask(id, image, question, eval_mode)
```

**Message shape.** One user turn: the image and `task.question`. No system prompt, no few-shot
examples, no extra instructions. The prompts already say how to format the answer.

**Decoding.** Follow the protocol in the DatBench paper: up to 4,096 new tokens and the sampling
settings recommended by the model card (greedy if the card gives none). Use the same settings for
every model you compare.

**Answer formats.** The prompts state the expected format. Scorers depend on it:

| Rows | Expected answer |
|---|---|
| Most short-answer and multiple-choice rows | `\boxed{...}`; the last `\boxed{}` in the output is used |
| Grounding boxes (RefCOCO family) | `\boxed{[x1, y1, x2, y2]}` on a 0-1000 scale (0-1 also accepted). Absolute pixel coordinates are mis-scored |
| Pointing (PixMo) | `\boxed{<x, y>}` on a 0-1000 scale |
| Key-information extraction (CC-OCR KIE, OCRBench v2 KIE) | raw JSON, no `\boxed{}` |

## 4. Package the outputs

```python
import re
from datbench import VLMResponse

THINK = re.compile(r"\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

def clean(text: str) -> str:
    return THINK.sub("", text or "").strip()

vlm_responses = [VLMResponse(id=t.id, raw_output=clean(outputs[t.id])) for t in tasks]
```

- Strip reasoning traces yourself. The scorers take the *last* `\boxed{}` anywhere in the text, so a
  box inside a think block can override the real answer. Only the RealWorldQA scorer strips think
  blocks itself.
- Leave `parsed_answer` unset. If set, it replaces `raw_output` for both the scorers and the judge.
- Send one response per task. Rows without a response are skipped and shrink the denominator.

## 5. Judge step

Judge-mode rows at the pinned revision:

| Capability | Judge rows | Source |
|---|---|---|
| chart | 1,984 of 5,000 | CharXiv descriptive and reasoning |
| general | 2,620 of 5,000 | VQA-v2 |
| math | 3,640 of 5,000 | MathVerse |
| table | 1,857 of 5,000 | AI2D (open-ended) |
| counting, document, grounding, scene, spatial | 0 | |

`evaluator.create_judge_tasks(vlm_responses)` returns one `JudgeRequest(id, vlm_response,
ground_truth, judge_prompt)` per judge-mode row. `judge_prompt` is the row's rubric; when the
rubric ends with a two-line `Reason: / answer:` instruction it is rewritten to ask for a single
boolean.

What the rubric does **not** give you:

- Placeholders such as `{question}`, `{correct_answer}`, `{model_answer}` and
  `{overarching_question}` are never filled. `str.format` fails on several rubrics because they
  contain literal JSON braces.
- `ground_truth` is the single primary answer. VQA-v2 rubrics expect all human answers, which are
  in `evaluator.samples_by_id[id].all_answers`.
- The CharXiv descriptive rubrics have no verdict format of their own.

### Reference judge setup

Our reported numbers use this setup. Use it to stay comparable:

- Decoding: temperature 0, top-p 1, max 256 new tokens.
- Messages: the rubric, then a JSON payload with the formatted question, the references and the
  model's final answer, then an explicit output-format line:

```python
import json
from datbench.datasets_scoring.evaluation_utils.erma_utils import extract_final_answer

JUDGE_SYSTEM = "You are a strict evaluator. Follow the provided rubric exactly and return only a final boolean decision."
JUDGE_FORMAT = (
    "Output format requirement (must follow exactly):\n"
    '{"answer": true|false}\n'
    "Do not output anything else. No reason field, no explanation, no extra keys."
)

def judge_messages(evaluator, task_by_id, request, capability):
    sample = evaluator.samples_by_id[request.id]
    raw = request.vlm_response.strip()
    payload = {
        "question": task_by_id[request.id].question.strip(),
        "ground_truth": request.ground_truth,
        "accepted_answers": list(sample.all_answers),
        "model_response": extract_final_answer(raw).strip() or raw,
        "context": {
            "sample_id": request.id,
            "capability": capability,
            "source_info": dict(sample.source_info),
            "eval_metrics": dict(sample.eval_metrics),
            "metadata": json.loads(sample.metadata or "{}"),
        },
    }
    user = (
        f"{request.judge_prompt}\n\nEvaluate the following sample payload:\n"
        f"{json.dumps(payload, ensure_ascii=True)}\n\n{JUDGE_FORMAT}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
```

`model_response` is the extracted final answer (the last `\boxed{}`, else an `Answer:` line, else
the last non-empty line), not the full reasoning. If a model's final answer is very long, shorten
only `model_response` so the message fits the judge's context window.

### Verdicts

```python
from datbench import JudgeResponse, parse_judge_verdict

judge_responses = [JudgeResponse(id=r.id, raw_judge_output=clean(judge_text[r.id])) for r in judge_requests]
assert all(parse_judge_verdict(None, r.raw_judge_output) is not None for r in judge_responses)
```

`parse_judge_verdict` accepts `{"answer": true|false}` (optionally in a code fence), an
`answer: true|false` line, or a bare `true` / `false` / `yes` / `no` / `correct` / `incorrect`.
It does not read `score` keys, so strip think blocks from reasoning judges. If you parse verdicts
yourself, pass `verdict="correct"` or `"incorrect"`; it takes precedence over the raw text.

## 6. Score and read the report

```python
report = evaluator.compute_metrics(vlm_responses, judge_responses=judge_responses, strict_judge=True)
score = report.summary["overall_accuracy"]
report.save(f"datbench_{capability}.json")
```

- Direct rows are scored by the scorer of their source dataset (`source_info["dataset"]`). Judge
  rows score 1.0 or 0.0 from the verdict.
- `overall_accuracy` (equal to `capability_accuracy[capability]`) is the headline number for the
  capability. Do not recompute it from `is_correct` or `total_correct`: those threshold or truncate
  fractional scores.
- `dataset_metrics` holds each source's own diagnostics (e.g. raw ANLS); they are not averaged into
  the headline.
- One evaluator scores one capability. If you report a single DatBench number, use the unweighted
  mean of the nine capability scores, and report the nine scores alongside it.

## Checklist

Before trusting a capability score:

- [ ] `report.summary["total_samples"] == len(evaluator.samples)`. Rows without a response are skipped silently.
- [ ] `judge_missing_responses == 0` and `judge_unparsed_verdicts == 0` (judge capabilities).
- [ ] The dataset revision and package commit are recorded with the score.
- [ ] `nltk` is below 3.10.3 and `wordnet` loads (`document`).
- [ ] Outputs had think blocks stripped.
- [ ] Boxes and points are on the 0-1000 scale (`grounding`).

## Memory and throughput

`DatBenchEvaluator` decodes and keeps every image when constructed: about 12 GB peak for `math` and
tens of GB for `chart`, `document` or `scene`. Scoring never reads images, so a low-memory setup
splits generation and scoring:

```python
from datasets import Image

for row in dataset:                       # generation: one decoded image at a time
    prompt = row["prompt_format"]["prefix"] + row["question"] + row["prompt_format"]["suffix"]
    outputs[row["id"]] = my_vlm(row["image"], prompt)

scorer = DatBenchEvaluator(dataset.cast_column("image", Image(decode=False)), capability)
```

The prompt built this way is identical to `task.question`. On `math` this lowers peak memory from
about 12 GB to about 1 GB with identical scores. Score one capability per process for DatBench-Full,
whose `general` config has 59,610 rows (55,056 of them judge rows).

## End-to-end example

Uses `load_capability`, `clean` and `judge_messages` from the sections above.

```python
def evaluate(capability, run_vlm, run_judge):
    """run_vlm(image, prompt) -> str; run_judge(messages) -> str."""
    dataset = load_capability(capability)
    evaluator = DatBenchEvaluator(dataset, capability)
    tasks = evaluator.get_inference_tasks()
    task_by_id = {t.id: t for t in tasks}

    vlm_responses = [VLMResponse(id=t.id, raw_output=clean(run_vlm(t.image, t.question))) for t in tasks]
    judge_responses = [
        JudgeResponse(id=r.id, raw_judge_output=clean(run_judge(judge_messages(evaluator, task_by_id, r, capability))))
        for r in evaluator.create_judge_tasks(vlm_responses)
    ]
    report = evaluator.compute_metrics(vlm_responses, judge_responses=judge_responses, strict_judge=True)

    summary = report.summary
    assert summary["total_samples"] == len(evaluator.samples)
    assert summary["judge_missing_responses"] == 0 and summary["judge_unparsed_verdicts"] == 0
    return summary["overall_accuracy"], report

scores = {cap: evaluate(cap, my_vlm, my_judge)[0] for cap in CAPABILITIES}
datbench_score = sum(scores.values()) / len(scores)
```

In practice `run_vlm` should batch requests and retry failures; the loop above is sequential only
for clarity.
