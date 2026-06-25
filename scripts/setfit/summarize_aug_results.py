import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
GENERATED_FILES = {"raw_results.csv", "summary_results.csv", "comparison_results.csv"}
METHOD_ORDER = ["vanilla", "random_aug", "error_driven"]
METHOD_ALIASES = {
    "vanilla": "vanilla",
    "baseline": "vanilla",
    "random_aug": "random_aug",
    "random": "random_aug",
    "random_augmentation": "random_aug",
    "random augmentation": "random_aug",
    "augmented": "error_driven",
    "error_driven": "error_driven",
    "error-driven": "error_driven",
    "error_driven_aug": "error_driven",
    "error-driven augmentation": "error_driven",
}
RAW_FIELDS = [
    "dataset",
    "sample_size",
    "seed",
    "iteration",
    "split",
    "method",
    "accuracy",
    "macro_f1",
    "delta_vs_vanilla",
    "delta_vs_random_aug",
    "per_class_accuracy_delta",
    "result_file",
]
SUMMARY_FIELDS = [
    "dataset",
    "sample_size",
    "split",
    "method",
    "n_runs",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "delta_error_driven_vs_vanilla",
    "delta_random_aug_vs_vanilla",
    "delta_error_driven_vs_random_aug",
]
COMPARISON_FIELDS = [
    "dataset",
    "sample_size",
    "split",
    "n_runs",
    "vanilla_accuracy_mean",
    "vanilla_accuracy_std",
    "random_aug_accuracy_mean",
    "random_aug_accuracy_std",
    "error_driven_accuracy_mean",
    "error_driven_accuracy_std",
    "random_aug_vs_vanilla_accuracy_delta",
    "error_driven_vs_vanilla_accuracy_delta",
    "error_driven_vs_random_aug_accuracy_delta",
    "vanilla_macro_f1_mean",
    "vanilla_macro_f1_std",
    "random_aug_macro_f1_mean",
    "random_aug_macro_f1_std",
    "error_driven_macro_f1_mean",
    "error_driven_macro_f1_std",
    "random_aug_vs_vanilla_macro_f1_delta",
    "error_driven_vs_vanilla_macro_f1_delta",
    "error_driven_vs_random_aug_macro_f1_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SetFit augmentation benchmark results.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"warning: {message}")


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def normalize_method(method: Any, source_file: Optional[Path] = None) -> Optional[str]:
    if method is not None and str(method).strip():
        key = str(method).strip().lower().replace(" ", "_")
        normalized = METHOD_ALIASES.get(key)
        if normalized:
            return normalized
    if source_file is not None:
        stem = source_file.stem.lower()
        if stem in METHOD_ALIASES:
            return METHOD_ALIASES[stem]
    return None


def normalize_row(record: Dict[str, Any], source_file: Path) -> Optional[Dict[str, Any]]:
    method = normalize_method(record.get("method"), source_file)
    dataset = record.get("dataset") or record.get("task")
    sample_size = record.get("sample_size", record.get("shot", record.get("shots")))
    accuracy = safe_float(record.get("accuracy", record.get("acc", record.get("score"))))
    macro_f1 = safe_float(record.get("macro_f1", record.get("macro-F1", record.get("macro_f1_score"))))

    if dataset is None or method is None or sample_size in (None, "") or accuracy is None:
        return None

    improvement = record.get("improvement_delta") if isinstance(record.get("improvement_delta"), dict) else {}
    per_class_delta = improvement.get("per_class_accuracy_delta", record.get("per_class_accuracy_delta", ""))
    if isinstance(per_class_delta, (dict, list)):
        per_class_delta = json.dumps(per_class_delta, ensure_ascii=False, sort_keys=True)

    return {
        "dataset": str(dataset),
        "sample_size": safe_int(sample_size),
        "seed": safe_int(record.get("seed", "")),
        "iteration": safe_int(record.get("iteration", record.get("iter", ""))),
        "split": record.get("split", ""),
        "method": method,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "delta_vs_vanilla": "",
        "delta_vs_random_aug": "",
        "per_class_accuracy_delta": per_class_delta or "",
        "result_file": str(source_file),
    }


def extract_json_records(payload: Any, source_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            rows.extend(extract_json_records(item, source_file))
        return rows
    if not isinstance(payload, dict):
        return rows

    row = normalize_row(payload, source_file)
    if row is not None:
        rows.append(row)
        return rows

    for key in ("results", "records", "runs", "items", "summary", "metrics"):
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            rows.extend(extract_json_records(value, source_file))
    return rows


def parse_json_file(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - summarizer should keep going.
        warn(f"skip unreadable json {path}: {exc}")
        return []
    return extract_json_records(payload, path)


def parse_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f_in:
            for line_number, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.extend(extract_json_records(json.loads(line), path))
                except json.JSONDecodeError as exc:
                    warn(f"skip malformed jsonl line {path}:{line_number}: {exc}")
    except OSError as exc:
        warn(f"skip unreadable jsonl {path}: {exc}")
    return rows


def parse_csv_file(path: Path) -> List[Dict[str, Any]]:
    if path.name in GENERATED_FILES:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for record in reader:
                row = normalize_row(record, path)
                if row is not None:
                    rows.append(row)
    except Exception as exc:  # noqa: BLE001
        warn(f"skip unreadable csv {path}: {exc}")
    return rows


def parse_text_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"Completed dataset=(?P<dataset>\S+) shot=(?P<shot>\d+) seed=(?P<seed>\d+): "
        r"vanilla=(?P<vanilla>[-+]?\d+(?:\.\d+)?), "
        r"random=(?P<random>[-+]?\d+(?:\.\d+)?), "
        r"error-driven=(?P<error>[-+]?\d+(?:\.\d+)?)"
    )
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        warn(f"skip unreadable text {path}: {exc}")
        return rows

    for match in pattern.finditer(text):
        common = {
            "dataset": match.group("dataset"),
            "sample_size": safe_int(match.group("shot")),
            "seed": safe_int(match.group("seed")),
            "iteration": "",
            "split": "",
            "macro_f1": None,
            "delta_vs_vanilla": "",
            "delta_vs_random_aug": "",
            "per_class_accuracy_delta": "",
            "result_file": str(path),
        }
        for method, group_name in (("vanilla", "vanilla"), ("random_aug", "random"), ("error_driven", "error")):
            row = dict(common)
            row["method"] = method
            row["accuracy"] = safe_float(match.group(group_name))
            rows.append(row)
    return rows


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        warn(f"results_dir does not exist: {results_dir}")
        return []
    suffixes = {".json", ".jsonl", ".csv", ".log", ".txt"}
    return sorted(path for path in results_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def read_all_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_result_files(results_dir):
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows.extend(parse_json_file(path))
        elif suffix == ".jsonl":
            rows.extend(parse_jsonl_file(path))
        elif suffix == ".csv":
            rows.extend(parse_csv_file(path))
        else:
            rows.extend(parse_text_file(path))
    return rows


def run_key(row: Dict[str, Any]) -> Tuple[str, Any, Any, Any, str]:
    run_id = row.get("seed") if row.get("seed") != "" else row.get("iteration", "")
    return (row["dataset"], row["sample_size"], run_id, row.get("split", ""), row["method"])


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, Any, Any, Any, str], Dict[str, Any]] = {}
    for row in rows:
        key = run_key(row)
        previous = best.get(key)
        if previous is None or previous["result_file"].endswith(".csv") and not row["result_file"].endswith(".csv"):
            best[key] = row
    return sorted(best.values(), key=lambda row: (row["dataset"], row["sample_size"], str(row.get("split", "")), str(row.get("seed", "")), str(row.get("iteration", "")), row["method"]))


def add_run_deltas(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, Any, Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        run_id = row.get("seed") if row.get("seed") != "" else row.get("iteration", "")
        grouped[(row["dataset"], row["sample_size"], run_id, row.get("split", ""))][row["method"]] = row

    for methods in grouped.values():
        vanilla = safe_float(methods.get("vanilla", {}).get("accuracy"))
        random_aug = safe_float(methods.get("random_aug", {}).get("accuracy"))
        for method, row in methods.items():
            accuracy = safe_float(row.get("accuracy"))
            if accuracy is not None and vanilla is not None:
                row["delta_vs_vanilla"] = accuracy - vanilla
            if accuracy is not None and random_aug is not None:
                row["delta_vs_random_aug"] = accuracy - random_aug


def values(rows: List[Dict[str, Any]], field: str) -> List[float]:
    return [value for value in (safe_float(row.get(field)) for row in rows) if value is not None]


def mean_or_blank(nums: List[float]) -> Any:
    return mean(nums) if nums else ""


def std_or_blank(nums: List[float]) -> Any:
    if not nums:
        return ""
    return stdev(nums) if len(nums) > 1 else 0.0


def build_summary_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Any, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["dataset"], row["sample_size"], row.get("split", ""), row["method"])].append(row)

    accuracy_means: Dict[Tuple[str, Any, str, str], Optional[float]] = {}
    for key, group in grouped.items():
        accuracy_means[key] = mean_or_blank(values(group, "accuracy"))

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, sample_size, split, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        acc = values(group, "accuracy")
        f1 = values(group, "macro_f1")
        vanilla = accuracy_means.get((dataset, sample_size, split, "vanilla"))
        random_aug = accuracy_means.get((dataset, sample_size, split, "random_aug"))
        error_driven = accuracy_means.get((dataset, sample_size, split, "error_driven"))
        summary_rows.append(
            {
                "dataset": dataset,
                "sample_size": sample_size,
                "split": split,
                "method": method,
                "n_runs": len(group),
                "accuracy_mean": mean_or_blank(acc),
                "accuracy_std": std_or_blank(acc),
                "macro_f1_mean": mean_or_blank(f1),
                "macro_f1_std": std_or_blank(f1),
                "delta_error_driven_vs_vanilla": delta(error_driven, vanilla),
                "delta_random_aug_vs_vanilla": delta(random_aug, vanilla),
                "delta_error_driven_vs_random_aug": delta(error_driven, random_aug),
            }
        )
    return summary_rows


def delta(left: Any, right: Any) -> Any:
    left_float = safe_float(left)
    right_float = safe_float(right)
    return "" if left_float is None or right_float is None else left_float - right_float


def build_comparison_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Any, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        grouped[(row["dataset"], row["sample_size"], row.get("split", ""))][row["method"]].append(row)

    comparison_rows: List[Dict[str, Any]] = []
    for (dataset, sample_size, split), methods in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "sample_size": sample_size,
            "split": split,
            "n_runs": max((len(group) for group in methods.values()), default=0),
        }
        for method in METHOD_ORDER:
            method_rows = methods.get(method, [])
            acc = values(method_rows, "accuracy")
            f1 = values(method_rows, "macro_f1")
            row[f"{method}_accuracy_mean"] = mean_or_blank(acc)
            row[f"{method}_accuracy_std"] = std_or_blank(acc)
            row[f"{method}_macro_f1_mean"] = mean_or_blank(f1)
            row[f"{method}_macro_f1_std"] = std_or_blank(f1)

        row["random_aug_vs_vanilla_accuracy_delta"] = delta(row.get("random_aug_accuracy_mean"), row.get("vanilla_accuracy_mean"))
        row["error_driven_vs_vanilla_accuracy_delta"] = delta(row.get("error_driven_accuracy_mean"), row.get("vanilla_accuracy_mean"))
        row["error_driven_vs_random_aug_accuracy_delta"] = delta(row.get("error_driven_accuracy_mean"), row.get("random_aug_accuracy_mean"))
        row["random_aug_vs_vanilla_macro_f1_delta"] = delta(row.get("random_aug_macro_f1_mean"), row.get("vanilla_macro_f1_mean"))
        row["error_driven_vs_vanilla_macro_f1_delta"] = delta(row.get("error_driven_macro_f1_mean"), row.get("vanilla_macro_f1_mean"))
        row["error_driven_vs_random_aug_macro_f1_delta"] = delta(row.get("error_driven_macro_f1_mean"), row.get("random_aug_macro_f1_mean"))
        comparison_rows.append(row)
    return comparison_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
    except PermissionError as exc:
        warn(f"could not write {path}; close the file if it is open and rerun summarization: {exc}")
        return False
    except OSError as exc:
        warn(f"could not write {path}: {exc}")
        return False
    return True


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.3f}"


def print_comparison_table(comparison_rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "sample_size",
        "vanilla_acc",
        "random_acc",
        "error_acc",
        "err-van",
        "err-rand",
    ]
    table_rows: List[List[str]] = []
    for row in comparison_rows:
        table_rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("sample_size", "")),
                fmt(row.get("vanilla_accuracy_mean")),
                fmt(row.get("random_aug_accuracy_mean")),
                fmt(row.get("error_driven_accuracy_mean")),
                fmt(row.get("error_driven_vs_vanilla_accuracy_delta")),
                fmt(row.get("error_driven_vs_random_aug_accuracy_delta")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(cells: List[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(render(row))


def print_table(summary_rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, Any, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[(row["dataset"], row["sample_size"], row.get("split", ""))][row["method"]] = row

    headers = ["dataset", "sample_size", "split", "vanilla", "random_aug", "error_driven", "err-van", "rand-van", "err-rand"]
    table_rows: List[List[str]] = []
    for (dataset, sample_size, split), methods in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        vanilla = methods.get("vanilla", {})
        random_aug = methods.get("random_aug", {})
        error_driven = methods.get("error_driven", {})
        table_rows.append(
            [
                str(dataset),
                str(sample_size),
                str(split),
                fmt(vanilla.get("accuracy_mean")),
                fmt(random_aug.get("accuracy_mean")),
                fmt(error_driven.get("accuracy_mean")),
                fmt(error_driven.get("delta_error_driven_vs_vanilla")),
                fmt(random_aug.get("delta_random_aug_vs_vanilla")),
                fmt(error_driven.get("delta_error_driven_vs_random_aug")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(cells: List[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(render(row))


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    raw_rows = deduplicate_rows(read_all_rows(results_dir))
    add_run_deltas(raw_rows)
    summary_rows = build_summary_rows(raw_rows)
    comparison_rows = build_comparison_rows(raw_rows)

    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_results.csv"
    comparison_path = output_dir / "comparison_results.csv"
    written_paths = [
        (raw_path, write_csv(raw_path, raw_rows, RAW_FIELDS)),
        (summary_path, write_csv(summary_path, summary_rows, SUMMARY_FIELDS)),
        (comparison_path, write_csv(comparison_path, comparison_rows, COMPARISON_FIELDS)),
    ]

    print(f"results_dir: {results_dir}")
    print(f"raw rows: {len(raw_rows)}")
    for path, did_write in written_paths:
        status = "wrote" if did_write else "skipped write"
        print(f"{status}: {path}")
    if summary_rows:
        print()
        print_table(summary_rows)
        print()
        print_comparison_table(comparison_rows)
    else:
        print("No structured benchmark results found.")


if __name__ == "__main__":
    main()
