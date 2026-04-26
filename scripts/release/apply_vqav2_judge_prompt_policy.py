"""Apply the VQA-V2 judge prompt policy to DatBench General parquet shards.

This release script rewrites only `general` rows whose source dataset is
`vqa-v2`. It preserves row order and all fields except `judge_prompt` for those
targeted rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from datbench.judge_policies.vqav2 import (
    LEGACY_VQA_V2_POLICY_MARKERS,
    VQA_V2_JUDGE_POLICY_MARKER,
    with_vqa_v2_semantic_judge_policy,
)


POLICY_MARKER = VQA_V2_JUDGE_POLICY_MARKER
DEFAULT_COMMIT_MESSAGE = (
    "Update DatBench General VQA-V2 semantic judge prompt policy"
)


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    revision: str
    expected_rows: int | None

    @property
    def name(self) -> str:
        return self.repo_id.rsplit("/", 1)[-1]


def parse_repo_spec(value: str) -> RepoSpec:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "--repo must have format <repo_id>:<revision>[:expected_rows]"
        )

    expected_rows = None
    if len(parts) == 3:
        expected_rows = int(parts[2])
    return RepoSpec(repo_id=parts[0], revision=parts[1], expected_rows=expected_rows)


def is_vqa_v2_source(source_info: object) -> bool:
    return isinstance(source_info, dict) and source_info.get("dataset") == "vqa-v2"


def policy_present(prompt: object) -> bool:
    return isinstance(prompt, str) and POLICY_MARKER in prompt


def any_vqa_v2_policy_marker_present(prompt: object) -> bool:
    if not isinstance(prompt, str):
        return False
    return any(
        marker in prompt
        for marker in (POLICY_MARKER, *LEGACY_VQA_V2_POLICY_MARKERS)
    )


def require_field(table: pa.Table, field_name: str, path: Path) -> None:
    if table.schema.get_field_index(field_name) == -1:
        raise ValueError(f"{path} is missing required field {field_name!r}")


def replace_column(table: pa.Table, name: str, values: list[object]) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    field = table.schema.field(field_index)
    column = pa.array(values, type=field.type)
    return table.set_column(field_index, field, column)


def rewrite_table(table: pa.Table, path: Path) -> tuple[pa.Table, dict[str, int]]:
    for field_name in ("id", "source_info", "eval_mode", "judge_prompt"):
        require_field(table, field_name, path)

    source_infos = table["source_info"].to_pylist()
    eval_modes = table["eval_mode"].to_pylist()
    old_prompts = table["judge_prompt"].to_pylist()

    new_eval_modes: list[object] = []
    new_prompts: list[object] = []
    target_rows = 0
    changed_rows = 0
    converted_eval_mode_rows = 0
    target_policy_rows = 0
    non_target_policy_rows = 0

    for source_info, eval_mode, old_prompt in zip(
        source_infos, eval_modes, old_prompts, strict=True
    ):
        if not is_vqa_v2_source(source_info):
            non_target_policy_rows += int(any_vqa_v2_policy_marker_present(old_prompt))
            new_eval_modes.append(eval_mode)
            new_prompts.append(old_prompt)
            continue

        target_rows += 1
        new_eval_mode = "judge"
        base_prompt = "" if old_prompt is None else str(old_prompt)
        new_prompt = with_vqa_v2_semantic_judge_policy(base_prompt)
        converted_eval_mode_rows += int(eval_mode != new_eval_mode)
        changed_rows += int(new_prompt != old_prompt or eval_mode != new_eval_mode)
        target_policy_rows += int(policy_present(new_prompt))
        new_eval_modes.append(new_eval_mode)
        new_prompts.append(new_prompt)

    rewritten = replace_column(table, "eval_mode", new_eval_modes)
    rewritten = replace_column(rewritten, "judge_prompt", new_prompts)
    validate_rewrite(table, rewritten, path, source_infos)

    return rewritten, {
        "rows": table.num_rows,
        "target_rows": target_rows,
        "changed_rows": changed_rows,
        "converted_eval_mode_rows": converted_eval_mode_rows,
        "target_policy_rows": target_policy_rows,
        "non_target_policy_rows": non_target_policy_rows,
    }


def validate_rewrite(
    old_table: pa.Table,
    new_table: pa.Table,
    path: Path,
    source_infos: list[object],
) -> None:
    if old_table.num_rows != new_table.num_rows:
        raise ValueError(f"{path} row count changed")
    if old_table.schema != new_table.schema:
        raise ValueError(f"{path} schema changed")

    allowed_target_fields = {"eval_mode", "judge_prompt"}
    for index, field in enumerate(old_table.schema):
        if field.name in allowed_target_fields:
            continue
        if not old_table.column(index).equals(new_table.column(index)):
            raise ValueError(f"{path} changed non-target field {field.name!r}")

    old_eval_modes = old_table["eval_mode"].to_pylist()
    new_eval_modes = new_table["eval_mode"].to_pylist()
    old_prompts = old_table["judge_prompt"].to_pylist()
    new_prompts = new_table["judge_prompt"].to_pylist()
    rows = zip(
        source_infos,
        old_eval_modes,
        new_eval_modes,
        old_prompts,
        new_prompts,
        strict=True,
    )
    for row_index, (
        source_info,
        old_eval_mode,
        new_eval_mode,
        old_prompt,
        new_prompt,
    ) in enumerate(rows):
        if is_vqa_v2_source(source_info):
            if new_eval_mode != "judge":
                raise ValueError(
                    f"{path} did not convert VQA-V2 row {row_index} to judge mode"
                )
            continue
        if old_eval_mode != new_eval_mode:
            raise ValueError(
                f"{path} changed non-VQA-V2 eval_mode at row {row_index}"
            )
        if old_prompt != new_prompt:
            raise ValueError(
                f"{path} changed non-VQA-V2 judge_prompt at row {row_index}"
            )
        if any_vqa_v2_policy_marker_present(new_prompt):
            raise ValueError(
                f"{path} has VQA-V2 judge policy marker on non-target row {row_index}"
            )


def resolve_general_shard_names(api: HfApi, spec: RepoSpec) -> list[str]:
    files = api.list_repo_files(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
    )
    shards = sorted(
        file_path
        for file_path in files
        if file_path.startswith("general/") and file_path.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"No general parquet shards found in {spec.repo_id}")
    return shards


def empty_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "target_rows": 0,
        "changed_rows": 0,
        "converted_eval_mode_rows": 0,
        "target_policy_rows": 0,
        "non_target_policy_rows": 0,
    }


def add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] += value


def rewrite_parquet_file(
    source_path: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, int]:
    parquet_file = pq.ParquetFile(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = empty_counts()

    with pq.ParquetWriter(output_path, parquet_file.schema_arrow) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=parquet_file.schema_arrow)
            rewritten, batch_counts = rewrite_table(table, source_path)
            add_counts(counts, batch_counts)
            writer.write_table(rewritten)

    return counts


def prepare_repo_candidate(
    spec: RepoSpec,
    output_root: Path,
    batch_size: int,
) -> dict[str, object]:
    api = HfApi()
    download_root = output_root / "downloads" / spec.name
    candidate_root = output_root / "candidates" / spec.name

    file_reports: list[dict[str, object]] = []
    totals = empty_counts()

    for shard_name in resolve_general_shard_names(api, spec):
        source_path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision,
                filename=shard_name,
                local_dir=download_root,
            )
        )
        output_path = candidate_root / shard_name
        counts = rewrite_parquet_file(source_path, output_path, batch_size)
        add_counts(totals, counts)
        file_reports.append(
            {
                "path": shard_name,
                **counts,
            }
        )

    if spec.expected_rows is not None and totals["rows"] != spec.expected_rows:
        raise ValueError(
            f"{spec.repo_id} expected {spec.expected_rows} rows, got {totals['rows']}"
        )
    if totals["target_rows"] == 0:
        raise ValueError(f"{spec.repo_id} has no VQA-V2 target rows")
    if totals["target_policy_rows"] != totals["target_rows"]:
        raise ValueError(
            f"{spec.repo_id} target policy coverage mismatch: "
            f"{totals['target_policy_rows']} / {totals['target_rows']}"
        )
    if totals["non_target_policy_rows"] != 0:
        raise ValueError(
            f"{spec.repo_id} has policy marker on {totals['non_target_policy_rows']} "
            "non-target rows"
        )

    return {
        "repo_id": spec.repo_id,
        "old_revision": spec.revision,
        "candidate_root": str(candidate_root),
        "totals": totals,
        "files": file_reports,
    }


def upload_candidate(repo_report: dict[str, object], commit_message: str) -> str:
    api = HfApi()
    candidate_root = Path(str(repo_report["candidate_root"]))
    api.upload_folder(
        folder_path=str(candidate_root),
        repo_id=str(repo_report["repo_id"]),
        repo_type="dataset",
        path_in_repo=".",
        allow_patterns=["general/*.parquet"],
        commit_message=commit_message,
    )
    return api.repo_info(str(repo_report["repo_id"]), repo_type="dataset").sha


def write_manifest(output_root: Path, manifest: dict[str, object]) -> Path:
    manifest_path = output_root / "validation" / "vqav2_judge_prompt_policy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Dataset repo spec: <repo_id>:<revision>[:expected_rows]",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--commit-message", default=DEFAULT_COMMIT_MESSAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_reports = [
        prepare_repo_candidate(parse_repo_spec(repo_spec), output_root, args.batch_size)
        for repo_spec in args.repo
    ]

    if args.upload:
        for repo_report in repo_reports:
            repo_report["new_revision"] = upload_candidate(
                repo_report, args.commit_message
            )

    manifest = {
        "commit_message": args.commit_message,
        "uploaded": args.upload,
        "policy_marker": POLICY_MARKER,
        "repos": repo_reports,
    }
    manifest_path = write_manifest(output_root, manifest)
    print(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
