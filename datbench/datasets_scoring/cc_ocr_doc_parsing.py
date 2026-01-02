"""CC-OCR Document Parsing scoring."""

import re
from typing import Any, Dict, List, Tuple


def score_sample(sample: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Score a single CC-OCR Document Parsing sample.

    Scoring methodology based on CC-OCR official evaluator:
    - Documents: LaTeX normalization + edit distance
    - Formulas: LaTeX normalization + edit distance
    - Molecular: SMILES normalization + edit distance
    - Tables: HTML normalization + TEDS (Tree Edit Distance)

    Args:
        sample: Original sample data (from self.data)
        model_output: Model's generated output

    Returns:
        Dictionary containing scoring results
    """
    # Extract from boxed format if present, otherwise use raw output
    extracted_output = _extract_boxed_or_raw(model_output)

    # Get ground truth and content type
    ground_truth = sample.get("answer") or sample.get("ground_truth_answer", "")

    # Try to get l2_category from multiple possible locations
    l2_category = (
        sample.get("l2_category") or
        sample.get("l2-category") or  # Original HF dataset uses hyphen
        "doc"  # Default fallback
    )

    # Route to appropriate scoring method based on content type
    if l2_category == "table":
        score, details = _score_table(extracted_output, ground_truth)
    elif l2_category == "formula":
        score, details = _score_formula(extracted_output, ground_truth)
    elif l2_category == "molecular":
        score, details = _score_molecular(extracted_output, ground_truth)
    else:  # doc or unknown
        score, details = _score_doc(extracted_output, ground_truth)

    return {
        "score": score,
        "l2_category": l2_category,
        "ground_truth": ground_truth,
        "model_output": model_output,
        **details,
    }


def _extract_boxed_or_raw(text: str) -> str:
    """Extract content from boxed formats if present."""
    if not isinstance(text, str):
        return ""

    # First try pipe box format
    pipe_box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    pipe_match = re.search(pipe_box_pattern, text, re.DOTALL)
    if pipe_match:
        return pipe_match.group(1)

    # Then try standard \boxed{...}
    boxed_pattern = r'\\boxed\{(.*)\}'
    boxed_match = re.search(boxed_pattern, text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1)

    return text


def _score_doc(pred: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    """Score document LaTeX content using edit distance."""
    pred_norm = _normalize_doc_latex(pred)
    gt_norm = _normalize_doc_latex(gt)

    edit_dist = _compute_edit_distance(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))

    if max_len == 0:
        score = 1.0 if pred_norm == gt_norm else 0.0
    else:
        score = 1.0 - (edit_dist / max_len)

    score = max(0.0, min(1.0, score))
    return score, {
        "edit_distance": edit_dist,
        "pred_length": len(pred_norm),
        "gt_length": len(gt_norm),
    }


def _score_formula(pred: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    """Score formula LaTeX content using edit distance."""
    pred_norm = _normalize_formula(pred)
    gt_norm = _normalize_formula(gt)

    edit_dist = _compute_edit_distance(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))

    if max_len == 0:
        score = 1.0 if pred_norm == gt_norm else 0.0
    else:
        score = 1.0 - (edit_dist / max_len)

    score = max(0.0, min(1.0, score))
    return score, {
        "edit_distance": edit_dist,
        "pred_length": len(pred_norm),
        "gt_length": len(gt_norm),
    }


def _score_molecular(pred: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    """Score molecular SMILES content using edit distance."""
    pred_norm = _normalize_molecular(pred)
    gt_norm = _normalize_molecular(gt)

    edit_dist = _compute_edit_distance(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))

    if max_len == 0:
        score = 1.0 if pred_norm == gt_norm else 0.0
    else:
        score = 1.0 - (edit_dist / max_len)

    score = max(0.0, min(1.0, score))
    return score, {
        "edit_distance": edit_dist,
        "pred_length": len(pred_norm),
        "gt_length": len(gt_norm),
    }


def _score_table(pred: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    """Score table HTML content using TEDS (Tree Edit Distance)."""
    pred_norm = _normalize_table_html(pred)
    gt_norm = _normalize_table_html(gt)

    try:
        # Try to use TEDS if available
        score = _compute_teds(pred_norm, gt_norm)
        method = "teds"
    except ImportError:
        # Fallback to edit distance if TEDS dependencies not available
        edit_dist = _compute_edit_distance(pred_norm, gt_norm)
        max_len = max(len(pred_norm), len(gt_norm))
        if max_len == 0:
            score = 1.0 if pred_norm == gt_norm else 0.0
        else:
            score = 1.0 - (edit_dist / max_len)
        method = "edit_distance_fallback"

    score = max(0.0, min(1.0, score))
    return score, {
        "method": method,
        "pred_length": len(pred_norm),
        "gt_length": len(gt_norm),
    }


def _normalize_doc_latex(text: str) -> str:
    """Normalize document LaTeX content."""
    if not isinstance(text, str):
        return ""

    # Remove markdown code blocks
    text = re.sub(r'```latex\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)

    # Remove LaTeX package declarations and document structure
    patterns = [
        r'\\documentclass\{.*?\}',
        r'\\usepackage\{.*?\}',
        r'\\begin\{document\}',
        r'\\end\{document\}',
        r'\\geometry\{.*?\}',
        r'\\title\{.*?\}',
        r'\\author\{.*?\}',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text)

    # Strip all whitespace
    text = text.replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')
    return text


def _normalize_formula(text: str) -> str:
    """Normalize formula LaTeX content."""
    if not isinstance(text, str):
        return ""

    # Remove markdown code blocks
    text = re.sub(r'```latex\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)

    # Strip whitespace
    text = text.replace('\t', '').replace('\n', '').replace(' ', '')
    return text


def _normalize_molecular(text: str) -> str:
    """Normalize molecular SMILES content."""
    if not isinstance(text, str):
        return ""

    # Remove SMILES tags
    text = text.replace('<smiles>', '').replace('</smiles>', '')

    # Strip whitespace
    text = text.replace('\n', '').replace(' ', '')
    return text


def _normalize_table_html(text: str) -> str:
    """Normalize table HTML content."""
    if not isinstance(text, str):
        return ""

    # Extract table content
    table_match = re.search(r'<table>(.*?)</table>', text, re.DOTALL | re.IGNORECASE)
    if table_match:
        text = '<table>' + table_match.group(1) + '</table>'

    # Remove attributes from table opening tag
    text = re.sub(r'<table[^>]*>', '<table>', text, flags=re.IGNORECASE)

    # Collapse whitespace between tags
    text = re.sub(r'>\s+<', '><', text)

    # Strip newlines
    text = text.replace('\n', '')

    # Convert fullwidth to halfwidth
    text = _convert_to_halfwidth(text)

    return text


def _convert_to_halfwidth(text: str) -> str:
    """Convert fullwidth characters to halfwidth."""
    result = []
    for char in text:
        code = ord(char)
        # Fullwidth space (U+3000) -> regular space
        if code == 0x3000:
            result.append(' ')
        # Fullwidth ASCII variants (U+FF01-FF5E) -> regular ASCII
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return ''.join(result)


def _compute_teds(pred_html: str, gt_html: str) -> float:
    """Compute TEDS (Tree Edit Distance based Similarity) for HTML tables."""
    try:
        from apted import APTED
        from apted.helpers import Tree
        from lxml import html as lxml_html
    except ImportError as e:
        raise ImportError(
            "TEDS requires apted and lxml. Install with: pip install apted lxml"
        ) from e

    def html_to_tree(html_str: str) -> Tree:
        """Convert HTML string to tree structure for APTED."""
        try:
            doc = lxml_html.fromstring(html_str)
            return _element_to_tree(doc)
        except Exception:
            return Tree('table')

    def count_nodes(tree: Tree) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in tree.children:
            count += count_nodes(child)
        return count

    # Parse HTML to trees
    pred_tree = html_to_tree(pred_html)
    gt_tree = html_to_tree(gt_html)

    # Compute tree edit distance
    distance = APTED(pred_tree, gt_tree).compute_edit_distance()

    # Normalize by number of nodes in ground truth
    n_nodes = count_nodes(gt_tree)
    if n_nodes == 0:
        return 0.0

    score = 1.0 - (float(distance) / n_nodes)
    return max(0.0, score)


def _element_to_tree(element) -> 'Tree':
    """Convert lxml element to apted Tree."""
    from apted.helpers import Tree

    # Create node with tag name
    node = Tree(element.tag)

    # Add children recursively
    for child in element:
        node.children.append(_element_to_tree(child))

    return node


def _compute_edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance using nltk or fallback."""
    try:
        import nltk
        return nltk.edit_distance(s1, s2)
    except ImportError:
        return _levenshtein_distance(s1, s2)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CC-OCR Document Parsing metrics."""
    if not results:
        return {"accuracy": 0.0}

    # For doc_parsing, top-level score is already correct
    scores = []
    for result in results:
        score_details = result.get("score_details", result)
        scores.append(score_details.get("score", 0.0))

    accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "accuracy": accuracy,
        "score": accuracy,
    }
