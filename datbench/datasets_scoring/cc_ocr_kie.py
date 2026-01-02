"""CC-OCR KIE (Key Information Extraction) scoring."""

import json
import re
from typing import Any, Dict, List, Tuple


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single CC-OCR KIE sample.

    Scoring methodology based on CC-OCR official evaluator:
    - F1 score (field-level, micro-averaged)
    - nTED (Normalized Tree Edit Distance)
    - Both scores computed on normalized JSON structures

    Args:
        sample: Original sample data
        model_output: Model's generated JSON output

    Returns:
        Dictionary containing scoring results
    """
    # Get ground truth
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer", "")

    # Extract JSON from model output (may be wrapped in markdown)
    pred_json = _extract_json(model_output)
    gt_json = _extract_json(ground_truth)

    # Compute both F1 and nTED scores
    f1_score = _compute_f1(pred_json, gt_json)
    nted_score = _compute_nted(pred_json, gt_json)

    # Average of both metrics (official evaluator reports both)
    combined_score = (f1_score + nted_score) / 2.0

    return {
        "score": combined_score,
        "f1_score": f1_score,
        "nted_score": nted_score,
        "l2_category": sample.get("l2_category", ""),
        "ground_truth": ground_truth,
        "model_output": model_output,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling markdown code blocks and boxed formats."""
    if not isinstance(text, str):
        return {}

    # First, try to extract from pipe box format
    pipe_box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    pipe_match = re.search(pipe_box_pattern, text, re.DOTALL)
    if pipe_match:
        text = pipe_match.group(1).strip()
    else:
        # Try to extract from \boxed{...} format
        boxed_marker = '\\boxed{'
        boxed_start = text.find(boxed_marker)
        if boxed_start != -1:
            start_idx = boxed_start + len(boxed_marker)
            brace_count = 1
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        text = text[start_idx:i].strip()
                        break

    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try to parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except:
                            pass
        return {}


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not isinstance(text, str):
        return ""

    # Fullwidth to halfwidth conversion
    text = _convert_to_halfwidth(text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _convert_to_halfwidth(text: str) -> str:
    """Convert fullwidth characters to halfwidth."""
    fullwidth = '！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～'
    halfwidth = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'

    trans = str.maketrans(fullwidth, halfwidth)
    text = text.translate(trans)
    text = text.replace('\u3000', ' ')  # Fullwidth space to regular space

    return text


def _normalize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dictionary values recursively."""
    if not isinstance(data, dict):
        return data

    normalized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            normalized[key] = _normalize_dict(value)
        elif isinstance(value, list):
            normalized[key] = [
                _normalize_dict(item) if isinstance(item, dict) else _normalize_text(str(item))
                for item in value
            ]
        elif isinstance(value, str):
            normalized[key] = _normalize_text(value)
        else:
            normalized[key] = value

    return normalized


def _flatten_dict(data: Dict[str, Any], parent_key: str = "") -> List[Tuple[str, str]]:
    """Flatten nested dict to list of (key_path, value) tuples."""
    flattened = []

    def _flatten(value, key=""):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                new_key = f"{key}.{child_key}" if key else child_key
                _flatten(child_value, new_key)
        elif isinstance(value, list):
            for item in value:
                _flatten(item, key)
        else:
            flattened.append((key, str(value)))

    _flatten(data, parent_key)
    return flattened


def _compute_f1(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> float:
    """Compute F1 score for KIE (field-level, micro-averaged)."""
    # Normalize both dicts
    pred_norm = _normalize_dict(pred_json)
    gt_norm = _normalize_dict(gt_json)

    # Flatten to (key, value) tuples
    pred_flat = _flatten_dict(pred_norm)
    gt_flat = _flatten_dict(gt_norm)

    # Convert to sets for matching
    pred_set = set(pred_flat)
    gt_set = set(gt_flat)

    # Count true positives, false positives, false negatives
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    # F1 = TP / (TP + (FP + FN) / 2)
    denominator = tp + (fp + fn) / 2.0 + 1e-6
    f1 = tp / denominator

    return min(1.0, max(0.0, f1))


def _compute_nted(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> float:
    """Compute normalized tree edit distance accuracy."""
    try:
        from zss import simple_distance, Node
    except ImportError:
        # Fallback to F1 if zss not available
        return _compute_f1(pred_json, gt_json)

    # Normalize both dicts
    pred_norm = _normalize_dict(pred_json)
    gt_norm = _normalize_dict(gt_json)

    # Convert dicts to trees
    pred_tree = _dict_to_tree(pred_norm)
    gt_tree = _dict_to_tree(gt_norm)
    empty_tree = Node("root")

    # Compute tree edit distance
    try:
        pred_distance = simple_distance(pred_tree, gt_tree)
        gt_distance = simple_distance(empty_tree, gt_tree)

        if gt_distance == 0:
            return 1.0 if pred_distance == 0 else 0.0

        # nTED accuracy
        normalized_distance = pred_distance / gt_distance
        score = 1.0 - normalized_distance

        return min(1.0, max(0.0, score))
    except Exception:
        return _compute_f1(pred_json, gt_json)


def _dict_to_tree(data: Any, label: str = "root") -> 'Node':
    """Convert dictionary to tree structure for nTED calculation."""
    try:
        from zss import Node
    except ImportError:
        raise ImportError("nTED requires zss library. Install with: pip install zss")

    # Create node with current label
    node = Node(str(label))

    if isinstance(data, dict):
        # Add children for each key-value pair
        for key, value in sorted(data.items()):
            child = _dict_to_tree(value, key)
            node.addkid(child)
    elif isinstance(data, list):
        # Add children for each list item
        for idx, item in enumerate(data):
            child = _dict_to_tree(item, f"{label}[{idx}]")
            node.addkid(child)
    else:
        # Leaf node - add value as child
        if data is not None and str(data).strip():
            value_node = Node(str(data))
            node.addkid(value_node)

    return node


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CC-OCR KIE metrics."""
    if not results:
        return {"f1_score": 0.0, "nted_score": 0.0, "accuracy": 0.0}

    # Extract individual metric scores from score_details
    f1_scores = []
    nted_scores = []
    combined_scores = []

    for result in results:
        score_details = result.get("score_details", result)
        f1_scores.append(score_details.get("f1_score", 0.0))
        nted_scores.append(score_details.get("nted_score", 0.0))
        combined_scores.append(score_details.get("score", 0.0))

    # Average metrics
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    nted_avg = sum(nted_scores) / len(nted_scores) if nted_scores else 0.0
    combined_avg = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0

    return {
        "f1_score": f1_avg,
        "nted_score": nted_avg,
        "accuracy": combined_avg,
        "score": combined_avg,
    }
