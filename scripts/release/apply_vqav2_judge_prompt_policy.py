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
from huggingface_hub import HfApi, snapshot_download

from datbench.judge_policies.vqav2 import with_vqa_v2_final_answer_policy


POLICY_MARKER = "Final-answer extraction policy for VQA-V2 semantic judging"
DEFAULT_COMMIT_MESSAGE = (
    "Add final-answer policy to DatBench General VQA-V2 judge prompts"
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

    new_prompts: list[object] = []
    target_rows = 0
    changed_rows = 0
    target_policy_rows = 0
    non_target_policy_rows = 0

    for source_info, eval_mode, old_prompt in zip(
        source_infos, eval_modes, old_prompts, strict=True
    ):
        if not is_vqa_v2_source(source_info):
            non_target_policy_rows += int(policy_present(old_prompt))
            new_prompts.append(old_prompt)
            continue

        target_rows += 1
        if eval_mode != "judge":
            raise ValueError(
                f"{path} has VQA-V2 target row with eval_mode={eval_mode!r}; "
                "expected 'judge'"
            )

        base_prompt = "" if old_prompt is None else str(old_prompt)
        new_prompt = with_vqa_v2_final_answer_policy(base_prompt)
        changed_rows += int(new_prompt != old_prompt)
        target_policy_rows += int(policy_present(new_prompt))
        new_prompts.append(new_prompt)

    rewritten = replace_column(table, "judge_prompt", new_prompts)
    validate_rewrite(table, rewritten, path, source_infos)

    return rewritten, {
        "rows": table.num_rows,
        "target_rows": target_rows,
        "changed_rows": changed_rows,
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

    judge_prompt_index = old_table.schema.get_field_index("judge_prompt")
    for index, field in enumerate(old_table.schema):
        if index == judge_prompt_index:
            continue
        if not old_table.column(index).equals(new_table.column(index)):
            raise ValueError(f"{path} changed non-target field {field.name!r}")

    old_prompts = old_table["judge_prompt"].to_pylist()
    new_prompts = new_table["judge_prompt"].to_pylist()
    for row_index, (source_info, old_prompt, new_prompt) in enumerate(
        zip(source_infos, old_prompts, new_prompts, strict=True)
    ):
        if is_vqa_v2_source(source_info):
            continue
        if old_prompt != new_prompt:
            raise ValueError(
                f"{path} changed non-VQA-V2 judge_prompt at row {row_index}"
            )


def resolve_general_shards(snapshot_root: Path) -> list[Path]:
    shards = sorted((snapshot_root / "general").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No general parquet shards found in {snapshot_root}")
    return shards


def prepare_repo_candidate(
    spec: RepoSpec,
    output_root: Path,
) -> dict[str, object]:
    download_root = output_root / "downloads" / spec.name
    candidate_root = output_root / "candidates" / spec.name
    snapshot_root = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            allow_patterns=["general/*.parquet"],
            local_dir=download_root,
        )
    )

    file_reports: list[dict[str, object]] = []
    totals = {
        "rows": 0,
        "target_rows": 0,
        "changed_rows": 0,
        "target_policy_rows": 0,
        "non_target_policy_rows": 0,
    }

    for source_path in resolve_general_shards(snapshot_root):
        relative_path = source_path.relative_to(snapshot_root)
        output_path = candidate_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(source_path)
        rewritten, counts = rewrite_table(table, source_path)
        pq.write_table(rewritten, output_path, compression="snappy")

        for key, value in counts.items():
            totals[key] += value
        file_reports.append(
            {
                "path": str(relative_path),
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
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--commit-message", default=DEFAULT_COMMIT_MESSAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_reports = [
        prepare_repo_candidate(parse_repo_spec(repo_spec), output_root)
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
