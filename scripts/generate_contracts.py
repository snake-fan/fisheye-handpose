#!/usr/bin/env python3
"""Generate dependency-free Python and TypeScript project-contract constants."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "contracts" / "project-contract-v1.json"
PYTHON_TARGETS = (
    PROJECT_ROOT / "src" / "fisheye_handpose" / "_generated_project_contract.py",
    PROJECT_ROOT / "deploy" / "mmpose-h20" / "_generated_project_contract.py",
    PROJECT_ROOT / "deploy" / "mmpose-h20" / "scripts" / "_generated_project_contract.py",
    PROJECT_ROOT
    / "deploy"
    / "mmpose-h20"
    / "worker"
    / "fisheye_h20_worker"
    / "_generated_project_contract.py",
)
STATIC_SCHEMA_BINDINGS = (
    (
        PROJECT_ROOT / "deploy" / "mmpose-h20" / "environment.json",
        "H20_ENVIRONMENT",
    ),
    (
        PROJECT_ROOT / "deploy" / "mmpose-h20" / "h20-executor.example.json",
        "H20_EXECUTOR",
    ),
    (
        PROJECT_ROOT / "deploy" / "mmpose-h20" / "mano-assets.example.json",
        "MANO_ASSETS",
    ),
    (
        PROJECT_ROOT / "deploy" / "mmpose-h20" / "model-assets.json",
        "MODEL_ASSETS",
    ),
)
TYPESCRIPT_TARGET = PROJECT_ROOT / "frontend" / "src" / "domain" / "projectContract.generated.ts"
IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ContractError(ValueError):
    """The canonical project contract is malformed or internally inconsistent."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty array")
    strings = [_string(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(strings)) != len(strings):
        raise ContractError(f"{label} must not contain duplicates")
    return strings


def _integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    integers: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractError(f"{label}[{index}] must be an integer")
        integers.append(item)
    return integers


def _identifier_map(value: Any, label: str) -> dict[str, str]:
    mapping = _object(value, label)
    if not mapping:
        raise ContractError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for key in sorted(mapping):
        if not isinstance(key, str) or IDENTIFIER.fullmatch(key) is None:
            raise ContractError(f"{label} keys must be uppercase identifiers")
        result[key] = _string(mapping[key], f"{label}.{key}")
    if len(set(result.values())) != len(result):
        raise ContractError(f"{label} values must be unique")
    return result


def _load_contract() -> dict[str, Any]:
    document = _object(json.loads(SOURCE_PATH.read_text(encoding="utf-8")), "contract")
    expected_keys = {
        "schema_version",
        "contract_version",
        "trace",
        "run",
        "fhp21",
        "schema_ids",
        "mapping_ids",
    }
    if set(document) != expected_keys:
        raise ContractError("contract must contain exactly the canonical top-level fields")
    if document["schema_version"] != "fisheye-handpose/project-contract/v1":
        raise ContractError("unsupported project contract schema_version")
    if document["contract_version"] != 1:
        raise ContractError("unsupported project contract version")

    trace = _object(document["trace"], "trace")
    if set(trace) != {"stages", "statuses"}:
        raise ContractError("trace must contain exactly stages and statuses")
    stages = _string_list(trace["stages"], "trace.stages")
    statuses = _string_list(trace["statuses"], "trace.statuses")
    run = _object(document["run"], "run")
    if set(run) != {"statuses"}:
        raise ContractError("run must contain exactly statuses")
    run_statuses = _string_list(run["statuses"], "run.statuses")
    for label, values in (
        ("trace.stages", stages),
        ("trace.statuses", statuses),
        ("run.statuses", run_statuses),
    ):
        if any(IDENTIFIER.fullmatch(value) is None for value in values):
            raise ContractError(f"{label} entries must be uppercase identifiers")

    fhp21 = _object(document["fhp21"], "fhp21")
    if set(fhp21) != {
        "schema_id",
        "landmark_names",
        "parents",
        "edges",
        "tip_indices",
    }:
        raise ContractError("fhp21 must contain exactly the canonical topology fields")
    _string(fhp21["schema_id"], "fhp21.schema_id")
    names = _string_list(fhp21["landmark_names"], "fhp21.landmark_names")
    parents = _integer_list(fhp21["parents"], "fhp21.parents")
    tips = _integer_list(fhp21["tip_indices"], "fhp21.tip_indices")
    if len(names) != 21 or len(parents) != 21:
        raise ContractError("fhp21 must contain exactly 21 names and parents")
    if parents[0] != -1:
        raise ContractError("fhp21 landmark 0 must be the only root")
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= index or (index > 0 and parent == -1):
            raise ContractError(f"fhp21 parent {parent} is invalid for landmark {index}")
    if tips != sorted(set(tips)) or any(index < 0 or index >= len(names) for index in tips):
        raise ContractError("fhp21 tip_indices must be sorted, unique, and in range")
    raw_edges = fhp21["edges"]
    if not isinstance(raw_edges, list):
        raise ContractError("fhp21.edges must be an array")
    edges: list[list[int]] = []
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ContractError(f"fhp21.edges[{index}] must contain two indices")
        values = _integer_list(edge, f"fhp21.edges[{index}]")
        edges.append(values)
    expected_edges = [[parent, child] for child, parent in enumerate(parents) if parent >= 0]
    if edges != expected_edges:
        raise ContractError("fhp21.edges must match the declared parent topology")

    document["trace"] = {"stages": stages, "statuses": statuses}
    document["run"] = {"statuses": run_statuses}
    document["fhp21"] = {
        "schema_id": fhp21["schema_id"],
        "landmark_names": names,
        "parents": parents,
        "edges": edges,
        "tip_indices": tips,
    }
    document["schema_ids"] = _identifier_map(document["schema_ids"], "schema_ids")
    document["mapping_ids"] = _identifier_map(document["mapping_ids"], "mapping_ids")
    return document


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _python_tuple(name: str, values: list[Any]) -> list[str]:
    lines = [f"{name} = ("]
    for value in values:
        if isinstance(value, list):
            lines.append(f"    ({value[0]}, {value[1]}),")
        elif isinstance(value, str):
            lines.append(f"    {_quoted(value)},")
        else:
            lines.append(f"    {value},")
    lines.append(")")
    return lines


def _render_python(contract: dict[str, Any]) -> str:
    trace = contract["trace"]
    run = contract["run"]
    fhp21 = contract["fhp21"]
    schema_ids = contract["schema_ids"]
    mapping_ids = contract["mapping_ids"]
    lines = [
        '"""Generated from contracts/project-contract-v1.json; do not edit."""',
        "",
        "from __future__ import annotations",
        "",
        f"PROJECT_CONTRACT_SCHEMA = {_quoted(contract['schema_version'])}",
        f"PROJECT_CONTRACT_VERSION = {contract['contract_version']}",
        "",
        *_python_tuple("TRACE_STAGE_VALUES", trace["stages"]),
        "",
        *_python_tuple("TRACE_STATUS_VALUES", trace["statuses"]),
        "",
        *_python_tuple("RUN_STATUS_VALUES", run["statuses"]),
        "",
        f"FHP21_SCHEMA_ID = {_quoted(fhp21['schema_id'])}",
        *_python_tuple("FHP21_NAMES", fhp21["landmark_names"]),
        *_python_tuple("FHP21_PARENTS", fhp21["parents"]),
        *_python_tuple("FHP21_EDGES", fhp21["edges"]),
        *_python_tuple("FHP21_TIP_INDICES", fhp21["tip_indices"]),
        "",
    ]
    for key, value in schema_ids.items():
        lines.append(f"{key}_SCHEMA = {_quoted(value)}")
    lines.extend(["", "SCHEMA_IDS = {"])
    for key in schema_ids:
        lines.append(f'    "{key}": {key}_SCHEMA,')
    lines.extend(["}", ""])
    for key, value in mapping_ids.items():
        lines.append(f"{key}_MAPPING_ID = {_quoted(value)}")
    lines.extend(["", "MAPPING_IDS = {"])
    for key in mapping_ids:
        lines.append(f'    "{key}": {key}_MAPPING_ID,')
    lines.extend(["}", "", "__all__ = ["])
    exported = [
        "FHP21_EDGES",
        "FHP21_NAMES",
        "FHP21_PARENTS",
        "FHP21_SCHEMA_ID",
        "FHP21_TIP_INDICES",
        "MAPPING_IDS",
        "PROJECT_CONTRACT_SCHEMA",
        "PROJECT_CONTRACT_VERSION",
        "RUN_STATUS_VALUES",
        "SCHEMA_IDS",
        "TRACE_STAGE_VALUES",
        "TRACE_STATUS_VALUES",
        *(f"{key}_SCHEMA" for key in schema_ids),
        *(f"{key}_MAPPING_ID" for key in mapping_ids),
    ]
    for name in sorted(exported):
        lines.append(f'    "{name}",')
    lines.extend(["]", ""])
    return "\n".join(lines)


def _render_typescript(contract: dict[str, Any]) -> str:
    trace = contract["trace"]
    run = contract["run"]
    fhp21 = contract["fhp21"]
    lines = [
        "// Generated from contracts/project-contract-v1.json; do not edit.",
        "",
        f"export const PROJECT_CONTRACT_SCHEMA = {_quoted(contract['schema_version'])} as const;",
        f"export const PROJECT_CONTRACT_VERSION = {contract['contract_version']} as const;",
        "",
        "export const TRACE_STAGES = [",
        *(f"  {_quoted(value)}," for value in trace["stages"]),
        "] as const;",
        "export type TraceStage = (typeof TRACE_STAGES)[number];",
        "",
        "export const TRACE_STATUSES = [",
        *(f"  {_quoted(value)}," for value in trace["statuses"]),
        "] as const;",
        "export type TraceStatus = (typeof TRACE_STATUSES)[number];",
        "",
        "export const RUN_STATUSES = [",
        *(f"  {_quoted(value)}," for value in run["statuses"]),
        "] as const;",
        "export type RunStatus = (typeof RUN_STATUSES)[number];",
        "",
        f"export const FHP21_SCHEMA_ID = {_quoted(fhp21['schema_id'])} as const;",
        "export const FHP21_NAMES = [",
        *(f"  {_quoted(value)}," for value in fhp21["landmark_names"]),
        "] as const;",
        "export const FHP21_PARENTS = [",
        *(f"  {value}," for value in fhp21["parents"]),
        "] as const;",
        "export const FHP21_EDGES = [",
        *(f"  [{edge[0]}, {edge[1]}]," for edge in fhp21["edges"]),
        "] as const;",
        "export const FHP21_TIP_INDICES = [",
        *(f"  {value}," for value in fhp21["tip_indices"]),
        "] as const;",
        "",
        "export const SCHEMA_IDS = {",
        *(f"  {key}: {_quoted(value)}," for key, value in contract["schema_ids"].items()),
        "} as const;",
        "",
        "export const MAPPING_IDS = {",
        *(f"  {key}: {_quoted(value)}," for key, value in contract["mapping_ids"].items()),
        "} as const;",
        "",
    ]
    return "\n".join(lines)


def _expected_outputs(contract: dict[str, Any]) -> dict[Path, str]:
    python_output = _render_python(contract)
    return {
        **{target: python_output for target in PYTHON_TARGETS},
        TYPESCRIPT_TARGET: _render_typescript(contract),
    }


def _validate_static_schema_bindings(contract: dict[str, Any]) -> None:
    schema_ids = contract["schema_ids"]
    for path, schema_key in STATIC_SCHEMA_BINDINGS:
        try:
            document = _object(
                json.loads(path.read_text(encoding="utf-8")),
                str(path.relative_to(PROJECT_ROOT)),
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot parse {path.relative_to(PROJECT_ROOT)}: {exc}") from exc
        expected = schema_ids[schema_key]
        if document.get("schema_version") != expected:
            raise ContractError(
                f"{path.relative_to(PROJECT_ROOT)} schema_version must be {expected!r}"
            )


def _check(outputs: dict[Path, str]) -> int:
    stale: list[Path] = []
    for path, expected in outputs.items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            actual = None
        if actual != expected:
            stale.append(path)
    if stale:
        for path in stale:
            print(f"stale generated contract: {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        print(
            "run: uv run --locked --no-editable python scripts/generate_contracts.py",
            file=sys.stderr,
        )
        return 1
    print("generated project contracts are current")
    return 0


def _write(outputs: dict[Path, str]) -> int:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"generated {path.relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a checked-in generated output differs from the canonical JSON",
    )
    args = parser.parse_args(argv)
    try:
        contract = _load_contract()
        _validate_static_schema_bindings(contract)
        outputs = _expected_outputs(contract)
    except (ContractError, json.JSONDecodeError, OSError) as error:
        print(f"project contract generation failed: {error}", file=sys.stderr)
        return 2
    return _check(outputs) if args.check else _write(outputs)


if __name__ == "__main__":
    raise SystemExit(main())
