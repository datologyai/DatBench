#!/usr/bin/env python3
"""Convert DatBench General VQA-v2 rows to semantic judge scoring.

This release tool rewrites only rows in the `general` config where
`source_info.dataset == "vqa-v2"`. It preserves row order and all fields except
`eval_mode` and `judge_prompt` on those targeted rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

from datasets import Dataset, Image, load_dataset

from datbench.judge_prompts import VQA_V2_SEMANTIC_JUDGE_PROMPT


RELEASES = {
    "DatologyAI/DatBench": {
        "expected_rows": 5000,
        "num_shards": 4,
        "output_name": "DatBench",
    },
    "DatologyAI/DatBench-Full": {
        "expected_rows": 59685,
        "num_shards": 55,
        "output_name": "DatBench-Full",
    },
}

UNCHANGED_TARGET_FIELDS = {
    "id",
    "question",
    "prompt_format",
    "answer",
    "all_answers",
    "source_info",
    "metadata",
    "eval_metrics",
    "is_circular",
}


def is_vqav2_general_row(row: Dict[str, Any]) -> bool:
    source_info = row.get("source_info") or {}
    return source_info.get("dataset") == "vqa-v2"


def convert_row(row: Dict[str, Any]) -> Dict[str, str]:
    if not is_vqav2_general_row(row):
        return {
            "eval_mode": row.get("eval_mode", "direct"),
            "judge_prompt": row.get("judge_prompt", ""),
        }
    return {
        "eval_mode": "judge",
        "judge_prompt": VQA_V2_SEMANTIC_JUDGE_PROMPT,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _field_value(row: Dict[str, Any], field: str) -> Any:
    return _jsonable(row.get(field))


def _counts_by_eval_mode(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("eval_mode", "direct")) for row in rows))


def _counts_by_source(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        source_info = row.get("source_info") or {}
        counts[str(source_info.get("dataset", ""))] += 1
    return dict(counts)


def validate_conversion(
    before: Dataset,
    after: Dataset,
    repo: str,
    revision: str,
) -> Dict[str, Any]:
    expected_rows = RELEASES[repo]["expected_rows"]
    if len(before) != expected_rows:
        raise ValueError(f"{repo} general row count changed before migration: {len(before)}")
    if len(after) != expected_rows:
        raise ValueError(f"{repo} general row count changed after migration: {len(after)}")

    changed_row_count = 0
    target_row_count = 0
    non_target_eval_mode_changes = []
    non_target_judge_prompt_changes = []
    target_field_diffs = Counter()
    empty_all_answers = []
    row_order_mismatches = []

    before_eval_counts = Counter()
    after_eval_counts = Counter()
    source_counts = Counter()

    for idx, (old, new) in enumerate(zip(before, after)):
        old_id = old["id"]
        new_id = new["id"]
        if old_id != new_id:
            row_order_mismatches.append({"idx": idx, "before": old_id, "after": new_id})
        before_eval_counts[str(old.get("eval_mode", "direct"))] += 1
        after_eval_counts[str(new.get("eval_mode", "direct"))] += 1
        source_info = old.get("source_info") or {}
        source_counts[str(source_info.get("dataset", ""))] += 1

        is_target = is_vqav2_general_row(old)
        if is_target:
            target_row_count += 1
            if not new.get("all_answers"):
                empty_all_answers.append(old_id)
            if new.get("eval_mode") != "judge":
                raise ValueError(f"Target row {old_id} was not converted to judge mode")
            if not new.get("judge_prompt"):
                raise ValueError(f"Target row {old_id} has empty judge_prompt")
            if old.get("eval_mode") != new.get("eval_mode") or old.get("judge_prompt") != new.get("judge_prompt"):
                changed_row_count += 1
            for field in UNCHANGED_TARGET_FIELDS:
                if _field_value(old, field) != _field_value(new, field):
                    target_field_diffs[field] += 1
            continue

        if old.get("eval_mode", "direct") != new.get("eval_mode", "direct"):
            non_target_eval_mode_changes.append(old_id)
        if old.get("judge_prompt", "") != new.get("judge_prompt", ""):
            non_target_judge_prompt_changes.append(old_id)

    if row_order_mismatches:
        raise ValueError(f"{repo} row order changed: {row_order_mismatches[:5]}")
    if non_target_eval_mode_changes:
        raise ValueError(
            f"{repo} non-target eval_mode changed: {non_target_eval_mode_changes[:5]}"
        )
    if non_target_judge_prompt_changes:
        raise ValueError(
            f"{repo} non-target judge_prompt changed: {non_target_judge_prompt_changes[:5]}"
        )
    if target_field_diffs:
        raise ValueError(f"{repo} target fields changed: {dict(target_field_diffs)}")
    if empty_all_answers:
        raise ValueError(f"{repo} target rows have empty all_answers: {empty_all_answers[:5]}")
    if changed_row_count != target_row_count:
        raise ValueError(
            f"{repo} changed rows ({changed_row_count}) != target rows ({target_row_count})"
        )

    return {
        "repo": repo,
        "old_revision": revision,
        "capability": "general",
        "split": "test",
        "row_count_before": len(before),
        "row_count_after": len(after),
        "changed_row_count": changed_row_count,
        "target_row_count": target_row_count,
        "source_dataset_counts": dict(source_counts),
        "eval_mode_counts_before": dict(before_eval_counts),
        "eval_mode_counts_after": dict(after_eval_counts),
        "field_level_diff_summary": {
            "target_changed_fields": ["eval_mode", "judge_prompt"],
            "target_unchanged_fields_checked": sorted(UNCHANGED_TARGET_FIELDS),
            "target_unchanged_fields_by_construction": {
                "image": "preserved by datasets.Dataset.map; validator does not decode image payloads"
            },
            "non_target_changed_fields": [],
        },
    }


def write_sharded_general(dataset: Dataset, repo: str, output_root: Path) -> Dict[str, Any]:
    release = RELEASES[repo]
    output_dir = output_root / release["output_name"] / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_idx in range(release["num_shards"]):
        shard = dataset.shard(
            num_shards=release["num_shards"],
            index=shard_idx,
            contiguous=True,
        )
        path = output_dir / f"test-{shard_idx:05d}-of-{release['num_shards']:05d}.parquet"
        shard.to_parquet(str(path))
        shard_paths.append(str(path))
    return {"output_dir": str(output_dir), "shards": shard_paths}


def migrate_release(repo: str, revision: str, output_root: Path) -> Dict[str, Any]:
    before = load_dataset(repo, "general", split="test", revision=revision)
    before = before.cast_column("image", Image(decode=False))
    after = before.map(convert_row, desc=f"Convert {repo} general VQA-v2 rows")
    validation = validate_conversion(
        before.remove_columns(["image"]),
        after.remove_columns(["image"]),
        repo=repo,
        revision=revision,
    )
    outputs = write_sharded_general(after, repo=repo, output_root=output_root)
    validation["candidate_output"] = outputs
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--datbench-revision", required=True)
    parser.add_argument("--datbench-full-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    reports = [
        migrate_release(
            "DatologyAI/DatBench",
            revision=args.datbench_revision,
            output_root=args.output_root,
        ),
        migrate_release(
            "DatologyAI/DatBench-Full",
            revision=args.datbench_full_revision,
            output_root=args.output_root,
        ),
    ]

    args.report_path.write_text(json.dumps({"reports": reports}, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(args.report_path), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
