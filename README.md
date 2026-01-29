# DatBench Evaluation Library

Official evaluation harness for DatBench, a high-fidelity vision-language benchmark with exact scoring implementations.


## Quick Start

```python
from datasets import load_dataset
from datbench import DatBenchEvaluator, VLMResponse

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

# Compute metrics
report = evaluator.compute_metrics(vlm_responses)

# View results
print(f"Accuracy: {report.summary['overall_accuracy']:.2%}")
print(f"Per-dataset: {report.summary['dataset_metrics']}")

# Save results
report.save("results.json")
```

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
    "answer": str,                    # Ground truth answer
    "all_answers": List[str],         # Alternative valid answers
    "eval_mode": "direct",            # "direct" or "judge"
    "is_circular": bool,              # Circular evaluation variant
    "metadata": str,                  # JSON string with dataset-specific metadata
    "source_info": {
        "dataset": str,               # Source dataset name
        "original_idx": str           # Original sample ID
    }
}
```

## API Reference

### DatBenchEvaluator

Main evaluation class.

**Methods:**
- `__init__(hf_dataset, capability)` - Initialize with HF dataset
- `get_inference_tasks()` - Get list of InferenceTask objects
- `create_judge_tasks(vlm_responses)` - Create judge evaluation tasks
- `compute_metrics(vlm_responses, judge_responses=None)` - Score and generate report

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

**DatBenchReport**: Final results
- `summary`: Dict with overall_accuracy, dataset_metrics, etc.
- `results`: List[SampleScore] with per-sample details
- `save(path)`: Save report to JSON file

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
