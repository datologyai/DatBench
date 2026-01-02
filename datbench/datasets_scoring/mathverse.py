"""MathVerse scoring.

NOTE: MathVerse uses LM-judge scoring only, not direct scoring.
"""

from typing import Any, Dict, List


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """MathVerse uses LM-judge scoring only."""
    raise NotImplementedError("MathVerse uses LM-judge scoring only.")


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Metrics are produced during LM-judge aggregation."""
    return {}
