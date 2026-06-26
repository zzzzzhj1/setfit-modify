#!/usr/bin/env python3
"""Audit result provenance without running, moving, or deleting experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_CSV_NAMES = {
    "raw_results.csv",
    "summary_results.csv",
    "summary_metrics.csv",
    "comparison_results.csv",
    "summary_table.csv",
}
ERROR_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "RuntimeError")
DATASET_ALIASES = {
    "ag-news": "ag_news",
    "ag_news": "ag_news",
    "sst2": "sst2",
    "sst-2": "sst2",
    "trec": "trec",
    "trec-qc": "trec",
    "imdb": "imdb",
}
METHOD_NAMES = {"vanilla", "random_aug", "error_driven", "augmented"}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "mtime_ts": stat.st_mtime,
    }


def normalize_dataset(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return DATASET_ALIASES.get(text.lower(), text)


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def unique_join(values: set[str], limit: int = 30) -> str:
    cleaned = sorted(v for v in values if v != "")
    if len(cleaned) > limit:
        return ";".join(cleaned[:limit]) + f";...(+{len(cleaned) - limit})"
    return ";".join(cleaned)


def parse_from_path(path: Path) -> dict[str, str]:
    parts = path.as_posix().split("/")
    parsed = {"dataset": "", "shot": "", "seed": "", "method": ""}
    for part in parts:
        key = DATASET_ALIASES.get(part.lower())
        if key:
            parsed["dataset"] = key
        shot_match = re.fullmatch(r"(\d+)-shot", part)
        if shot_match:
            parsed["shot"] = shot_match.group(1)
        seed_match = re.fullmatch(r"seed[-_]?(\d+)", part)
        if seed_match:
            parsed["seed"] = seed_match.group(1)
    if path.suffix == ".json" and path.stem in METHOD_NAMES:
        parsed["method"] = path.stem
    return parsed


def parse_log_name(path: Path) -> dict[str, str]:
    name = path.name
    parsed = {"dataset": "", "shot": "", "seed": ""}
    match = re.search(
        r"(?P<dataset>ag[_-]news|sst-?2|trec|imdb).*?shot(?P<shot>\d+).*?seed(?P<seed>\d+)",
        name,
        flags=re.IGNORECASE,
    )
    if match:
        parsed["dataset"] = normalize_dataset(match.group("dataset").replace("-", "_"))
        parsed["shot"] = match.group("shot")
        parsed["seed"] = match.group("seed")
    return parsed


def find_value(row: dict[str, Any], names: tuple[str, ...]) -> str:
    lower_to_key = {str(key).lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None:
            return clean_scalar(row.get(key))
    return ""


def flatten_json_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "rows", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                records = [item for item in value if isinstance(item, dict)]
                if records:
                    return records
        return [data]
    return []


def analyze_records(
    records: list[dict[str, Any]],
    fallback: dict[str, str],
) -> tuple[dict[str, set[str]], Counter[tuple[str, str, str]], bool]:
    values = {
        "dataset": set(),
        "shot": set(),
        "method": set(),
        "seed": set(),
    }
    run_keys: Counter[tuple[str, str, str]] = Counter()
    has_seed_field = False
    for row in records:
        dataset = normalize_dataset(find_value(row, ("dataset", "task", "task_name"))) or fallback["dataset"]
        shot = (
            find_value(row, ("shot", "sample_size", "shots", "k", "num_shots"))
            or fallback["shot"]
        )
        method = find_value(row, ("method", "strategy", "variant")) or fallback["method"]
        seed = find_value(row, ("seed", "random_seed"))
        if "seed" in {str(key).lower() for key in row} or "random_seed" in {str(key).lower() for key in row}:
            has_seed_field = True
        seed = seed or fallback["seed"]

        values["dataset"].add(dataset)
        values["shot"].add(shot)
        values["method"].add(method)
        values["seed"].add(seed)
        if dataset and shot and method:
            run_keys[(dataset, shot, method)] += 1
    return values, run_keys, has_seed_field


def analyze_csv(path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = file_info(path)
    fallback = parse_from_path(path)
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    values, run_keys, has_seed_field = analyze_records(rows, fallback)
    return {
        "kind": "csv",
        "path": rel(path, root),
        **info,
        "line_count": len(rows) + (1 if columns else 0),
        "columns": columns,
        "unique_dataset": values["dataset"],
        "unique_shot": values["shot"],
        "unique_method": values["method"],
        "unique_seed": values["seed"],
        "has_seed_field": has_seed_field,
        "run_keys": run_keys,
        "read_error": "",
    }, rows


def analyze_json(path: Path, root: Path) -> dict[str, Any]:
    info = file_info(path)
    fallback = parse_from_path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        records = flatten_json_records(data)
        columns = sorted({str(key) for record in records for key in record})
        values, run_keys, has_seed_field = analyze_records(records, fallback)
        line_count = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        read_error = ""
    except Exception as exc:  # noqa: BLE001 - audit should keep going on malformed files.
        records = []
        columns = []
        values = {"dataset": set(), "shot": set(), "method": set(), "seed": set()}
        run_keys = Counter()
        has_seed_field = False
        line_count = 0
        read_error = f"{type(exc).__name__}: {exc}"
    return {
        "kind": "json",
        "path": rel(path, root),
        **info,
        "line_count": line_count,
        "columns": columns,
        "unique_dataset": values["dataset"],
        "unique_shot": values["shot"],
        "unique_method": values["method"],
        "unique_seed": values["seed"],
        "has_seed_field": has_seed_field,
        "run_keys": run_keys,
        "read_error": read_error,
    }


def analyze_log(path: Path, root: Path) -> dict[str, Any]:
    info = file_info(path)
    parsed = parse_log_name(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    completed = "Completed dataset=" in text
    has_error = any(pattern in text for pattern in ERROR_PATTERNS)
    completed_runs: list[tuple[str, str, str]] = []
    for match in re.finditer(r"Completed dataset=(\S+) shot=(\d+) seed=(\d+)", text):
        completed_runs.append((normalize_dataset(match.group(1)), match.group(2), match.group(3)))
    return {
        "kind": "log",
        "path": rel(path, root),
        **info,
        "line_count": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "contains_completed": completed,
        "contains_error": has_error,
        "parsed_dataset": parsed["dataset"],
        "parsed_shot": parsed["shot"],
        "parsed_seed": parsed["seed"],
        "completed_runs": completed_runs,
    }


def iter_targets(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    csv_paths: list[Path] = []
    json_paths: list[Path] = []
    log_paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in TARGET_CSV_NAMES:
            csv_paths.append(path)
        elif path.suffix == ".json":
            json_paths.append(path)
        elif path.match("scripts/setfit/logs/*.log"):
            log_paths.append(path)
    return sorted(csv_paths), sorted(json_paths), sorted(log_paths)


def possibly_old_reason(row: dict[str, Any], newest_mtime: float) -> str:
    path = row["path"]
    reasons: list[str] = []
    if path.startswith("scripts/results/"):
        reasons.append("under scripts/results")
    if "/tmp/" in path or path.startswith("tmp/"):
        reasons.append("temporary path")
    if newest_mtime and newest_mtime - float(row["mtime_ts"]) > 7 * 24 * 60 * 60:
        reasons.append(">7 days older than newest scanned result")
    return "; ".join(reasons)


def is_result_like_file(row: dict[str, Any]) -> bool:
    path = str(row["path"])
    if path.startswith(("results/", "results_", "scripts/results/")):
        return True
    if row["kind"] == "csv":
        return Path(path).name in TARGET_CSV_NAMES
    return any(row[key] - {""} for key in ("unique_dataset", "unique_shot", "unique_method"))


def seed_status(
    aggregate: dict[tuple[str, str, str], set[str]],
    dataset: str,
    required_counts: tuple[int, ...],
) -> str:
    observed: dict[str, set[str]] = defaultdict(set)
    for (ds, shot, _method), seeds in aggregate.items():
        if ds == dataset:
            observed[shot].update(seed for seed in seeds if seed != "")
    if not observed:
        return f"{dataset}: no scanned seed-level results"
    parts = []
    for shot, seeds in sorted(observed.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        count = len(seeds)
        flags = [f"{required}=yes" if count >= required else f"{required}=no" for required in required_counts]
        parts.append(f"shot={shot}: seeds={count} ({', '.join(flags)})")
    return f"{dataset}: " + "; ".join(parts)


def compute_missing_runs(aggregate: dict[tuple[str, str, str], set[str]]) -> list[str]:
    by_dataset_shot: dict[tuple[str, str], int] = defaultdict(int)
    for (dataset, shot, _method), seeds in aggregate.items():
        by_dataset_shot[(dataset, shot)] = max(by_dataset_shot[(dataset, shot)], len(seeds))

    missing: list[str] = []
    for key, seeds in sorted(aggregate.items()):
        expected = by_dataset_shot[(key[0], key[1])]
        if expected > len(seeds):
            missing.append(
                f"{key[0]} shot={key[1]} method={key[2]}: runs={len(seeds)}, expected={expected}, "
                f"seeds={unique_join(seeds)}"
            )
    return missing


def write_csv_report(path: Path, file_rows: list[dict[str, Any]], log_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "path",
        "size_bytes",
        "mtime",
        "line_count",
        "columns",
        "unique_dataset",
        "unique_shot_or_sample_size",
        "unique_method",
        "unique_seed",
        "dataset_shot_method_n_runs",
        "has_seed_field",
        "missing_seed_field",
        "possibly_old",
        "old_reason",
        "contains_completed",
        "contains_error",
        "parsed_dataset",
        "parsed_shot",
        "parsed_seed",
        "read_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in file_rows:
            writer.writerow({
                "kind": row["kind"],
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "line_count": row["line_count"],
                "columns": ";".join(row["columns"]),
                "unique_dataset": unique_join(row["unique_dataset"]),
                "unique_shot_or_sample_size": unique_join(row["unique_shot"]),
                "unique_method": unique_join(row["unique_method"]),
                "unique_seed": unique_join(row["unique_seed"]),
                "dataset_shot_method_n_runs": "; ".join(
                    f"{ds}|{shot}|{method}={count}"
                    for (ds, shot, method), count in sorted(row["run_keys"].items())
                ),
                "has_seed_field": row["has_seed_field"],
                "missing_seed_field": not row["has_seed_field"],
                "possibly_old": bool(row.get("old_reason")),
                "old_reason": row.get("old_reason", ""),
                "contains_completed": "",
                "contains_error": "",
                "parsed_dataset": "",
                "parsed_shot": "",
                "parsed_seed": "",
                "read_error": row["read_error"],
            })
        for row in log_rows:
            writer.writerow({
                "kind": "log",
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "line_count": row["line_count"],
                "columns": "",
                "unique_dataset": "",
                "unique_shot_or_sample_size": "",
                "unique_method": "",
                "unique_seed": "",
                "dataset_shot_method_n_runs": "",
                "has_seed_field": "",
                "missing_seed_field": "",
                "possibly_old": bool(row.get("old_reason")),
                "old_reason": row.get("old_reason", ""),
                "contains_completed": row["contains_completed"],
                "contains_error": row["contains_error"],
                "parsed_dataset": row["parsed_dataset"],
                "parsed_shot": row["parsed_shot"],
                "parsed_seed": row["parsed_seed"],
                "read_error": "",
            })


def write_text_report(
    path: Path,
    root: Path,
    file_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    aggregate: dict[tuple[str, str, str], set[str]],
    missing_runs: list[str],
) -> None:
    lines: list[str] = []
    lines.append("Result Provenance Audit")
    lines.append(f"Project root: {root}")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Scope")
    lines.append(f"- CSV/JSON files scanned: {len(file_rows)}")
    lines.append(f"- SetFit log files scanned: {len(log_rows)}")
    lines.append("- This script only scans files and writes audit_results_report.txt/csv.")
    lines.append("")
    lines.append("Seed Coverage Flags")
    lines.append(f"- {seed_status(aggregate, 'trec', (20,))}")
    lines.append(f"- {seed_status(aggregate, 'sst2', (5, 10))}")
    lines.append(f"- {seed_status(aggregate, 'ag_news', (5, 10))}")
    lines.append("")
    lines.append("Possibly Old Result Files")
    old_rows = [row for row in file_rows + log_rows if row.get("old_reason")]
    if old_rows:
        for row in old_rows:
            lines.append(f"- {row['path']} ({row['old_reason']})")
    else:
        lines.append("- none flagged")
    lines.append("")
    lines.append("Files Missing Seed Field")
    missing_seed = [row for row in file_rows if is_result_like_file(row) and not row["has_seed_field"]]
    if missing_seed:
        for row in missing_seed:
            lines.append(f"- {row['path']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Dataset-Shot-Method Missing Runs")
    if missing_runs:
        for item in missing_runs:
            lines.append(f"- {item}")
    else:
        lines.append("- none detected by observed-max heuristic")
    lines.append("")
    lines.append("CSV/JSON Details")
    for row in file_rows:
        lines.append(f"- {row['path']}")
        lines.append(f"  kind={row['kind']} size={row['size_bytes']} mtime={row['mtime']} lines={row['line_count']}")
        lines.append(f"  columns={';'.join(row['columns'])}")
        lines.append(
            "  unique "
            f"dataset={unique_join(row['unique_dataset']) or '-'} "
            f"shot/sample_size={unique_join(row['unique_shot']) or '-'} "
            f"method={unique_join(row['unique_method']) or '-'} "
            f"seed={unique_join(row['unique_seed']) or '-'}"
        )
        if row["run_keys"]:
            n_runs = "; ".join(
                f"{ds}|{shot}|{method}={count}"
                for (ds, shot, method), count in sorted(row["run_keys"].items())
            )
            lines.append(f"  dataset-shot-method n_runs: {n_runs}")
        if row["read_error"]:
            lines.append(f"  read_error={row['read_error']}")
    lines.append("")
    lines.append("Log Details")
    for row in log_rows:
        lines.append(f"- {row['path']}")
        lines.append(f"  size={row['size_bytes']} mtime={row['mtime']} lines={row['line_count']}")
        lines.append(
            f"  contains_completed={row['contains_completed']} contains_error={row['contains_error']} "
            f"parsed_dataset={row['parsed_dataset'] or '-'} parsed_shot={row['parsed_shot'] or '-'} "
            f"parsed_seed={row['parsed_seed'] or '-'}"
        )
        if row["completed_runs"]:
            completed = "; ".join(f"{ds}|{shot}|seed={seed}" for ds, shot, seed in row["completed_runs"])
            lines.append(f"  completed_runs={completed}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root to scan")
    parser.add_argument("--txt", type=Path, default=Path("audit_results_report.txt"))
    parser.add_argument("--csv", type=Path, default=Path("audit_results_report.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    csv_paths, json_paths, log_paths = iter_targets(root)

    file_rows: list[dict[str, Any]] = []
    seed_aggregate: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for csv_path in csv_paths:
        row, raw_rows = analyze_csv(csv_path, root)
        file_rows.append(row)
        for raw in raw_rows:
            fallback = parse_from_path(csv_path)
            dataset = normalize_dataset(find_value(raw, ("dataset", "task", "task_name"))) or fallback["dataset"]
            shot = find_value(raw, ("shot", "sample_size", "shots", "k", "num_shots")) or fallback["shot"]
            method = find_value(raw, ("method", "strategy", "variant")) or fallback["method"]
            seed = find_value(raw, ("seed", "random_seed")) or fallback["seed"]
            if dataset and shot and method and seed:
                seed_aggregate[(dataset, shot, method)].add(seed)

    for json_path in json_paths:
        row = analyze_json(json_path, root)
        file_rows.append(row)
        for dataset, shot, method in row["run_keys"]:
            for seed in row["unique_seed"]:
                if seed:
                    seed_aggregate[(dataset, shot, method)].add(seed)

    log_rows = [analyze_log(path, root) for path in log_paths]
    for log_row in log_rows:
        for dataset, shot, seed in log_row["completed_runs"]:
            for method in ("vanilla", "random_aug", "error_driven"):
                seed_aggregate[(dataset, shot, method)].add(seed)

    newest_mtime = max([row["mtime_ts"] for row in file_rows + log_rows], default=0.0)
    for row in file_rows + log_rows:
        row["old_reason"] = possibly_old_reason(row, newest_mtime)

    missing_runs = compute_missing_runs(seed_aggregate)
    write_csv_report((root / args.csv).resolve() if not args.csv.is_absolute() else args.csv, file_rows, log_rows)
    write_text_report(
        (root / args.txt).resolve() if not args.txt.is_absolute() else args.txt,
        root,
        file_rows,
        log_rows,
        seed_aggregate,
        missing_runs,
    )
    print(f"Wrote {args.txt} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
