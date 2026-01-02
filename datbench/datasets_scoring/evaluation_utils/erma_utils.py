"""Utilities for extracting concise answers from ERMA or CoT outputs.

This aims to reliably pull the final short answer for scoring, preferring the
explicit "ANSWER: ..." section when present, and falling back to common CoT
markers or the last short non-empty line.
"""

from __future__ import annotations

import re
from typing import Optional

_ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+)$")
_BOXED_RE = re.compile(r"\\+boxed\{([^}]+)\}")
_PIPE_BOX_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)


def _unwrap_known_tags(ans: str) -> str:
    """Unwrap common XML-like wrappers used by some models, preserving inner text.

    Examples:
    - "<answer>female</answer>" -> "female"
    - "<final_answer>42</final_answer>" -> "42"
    - "<final>Paris</final>" -> "Paris"
    """
    s = ans.strip()
    # Try a small set of tag names we see in practice
    tag_names = ["answer", "final_answer", "final"]
    for name in tag_names:
        pattern = rf"^<\s*{name}\s*>\s*(.*?)\s*<\s*/\s*{name}\s*>$"
        m = re.match(pattern, s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return s


def _is_placeholder(ans: str) -> bool:
    """Return True if `ans` is a placeholder (no real content).

    We treat angle-bracket placeholders and near-variants as invalid answers.
    """
    if not ans:
        return True
    s = ans.strip().lower()
    # Exact placeholders we want to invalidate
    placeholders = {
        "<answer>",
        "<final>",
        "<final answer>",
        "<final_answer>",
    }
    if s in placeholders:
        return True
    # Generic pattern like "< answer >"
    if re.fullmatch(r"<\s*answer\s*>", s):
        return True
    return False


def extract_final_answer(text: str) -> str:
    """Extract the final short answer from ERMA/CoT-style text.

    Strategy (in priority order):
    1. Extract content from \\boxed{...} format (highest priority)
    2. Prefer the last line starting with "ANSWER:" or "Final Answer:" (case-insensitive).
       Return the remainder of that line trimmed.
    3. Otherwise, take the last non-empty line if present.
    4. Strip simple wrappers (quotes) and collapse whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Helper to clean extracted candidates uniformly
    def _clean(ans: str) -> str:
        ans = (ans or "").strip()
        ans = ans.strip("\"'` ''")
        ans = _unwrap_known_tags(ans)
        ans = re.sub(r"\s+", " ", ans).strip()
        # Invalidate placeholder-only tokens
        if _is_placeholder(ans):
            return ""
        return ans

    # Priority 1: Check for boxed formats (handles \boxed{} and <|begin_of_box|>...<|end_of_box|>)
    # First check for pipe box format (used by some models like GLM)
    pipe_box_matches = list(_PIPE_BOX_RE.finditer(text))
    if pipe_box_matches:
        # Use the last occurrence (most likely the final answer)
        last_pipe_box = pipe_box_matches[-1]
        ans = _clean(last_pipe_box.group(1))
        return ans

    # Then check for standard \boxed{...} format
    boxed_matches = list(_BOXED_RE.finditer(text))
    if boxed_matches:
        # Use the last occurrence (most likely the final answer)
        last_boxed = boxed_matches[-1]
        ans = _clean(last_boxed.group(1))
        return ans

    # Priority 2: Prefer explicit ANSWER: line (last occurrence)
    last_match: Optional[re.Match] = None
    for m in _ANSWER_LINE_RE.finditer(text):
        last_match = m
    if last_match is not None:
        ans = _clean(last_match.group(1))
        return ans

    # Priority 3: Fallback: last non-empty line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    ans = _clean(lines[-1])
    return ans
