"""DatBench: High-Fidelity Vision-Language Evaluation.

This package provides the evaluation harness for DatBench datasets.
"""

from .schema import (
    DatBenchSample,
    InferenceTask,
    VLMResponse,
    JudgeRequest,
    JudgeResponse,
    SampleScore,
    DatBenchReport,
)
from .evaluator import DatBenchEvaluator

__version__ = "1.0.0"

__all__ = [
    "DatBenchSample",
    "InferenceTask",
    "VLMResponse",
    "JudgeRequest",
    "JudgeResponse",
    "SampleScore",
    "DatBenchReport",
    "DatBenchEvaluator",
]
