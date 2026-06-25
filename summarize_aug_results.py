import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


METHOD_ALIASES = {
    "vanilla": "vanilla",
    "baseline": "vanilla",
    "random_aug": "random_aug",
    "random_augmentation": "random_aug",
    "random augmentation": "random_aug",
    "augmented": "error_driven",
    "error_driven": "error_driven",
    "error-driven": "error_driven",
    "error_driven_aug": "error_driven",
    "error-driven augmentation": "error_driven",
}
METHODS = ["vanilla", "random_aug", "error_driven"]
RAW_FIELDS = [
    "dataset",
    "shot",
    "seed",
    "method",
    "accuracy",
    "macro_f1",
    "vanilla_accuracy",
    "random_aug_accuracy",
    "error_driven_accuracy",
    "accuracy_delta",
    "macro_f1_delta",
    "per_class_accuracy_delta",
    "source_file",
]
SUMMARY_FIELDS = [
    "dataset",
    "shot",
    "method",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_improvement_delta",
    "std_improvement_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SetFit augmentation benchmark JSON results.")
    parser.add_argument("--results_dir", default="results", help="Directory to recursively scan for JSON result files.")
    parser.add_argument("--output_dir", default=".", help="Directory where raw_results.csv and summary_results.csv are written.")
    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def normalize_method(method: Any) -> Optional[str]:
    if method is None:
        return None
    key = str(method).strip().lower().replace(" ", "_")
    return METHOD_ALIASES.get(key, key if key in METHODS else None)


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f_in:
            return json.load(f_in)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable JSON: {path} ({exc})")
        return None


def iter_candidate_json_files(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    return sorted(path for path in results_dir.rglob("*.json") if path.is_file())


def extract_records(payload: Any, source_file: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            records.extend(extract_records(item, source_file))
        return records

    if not isinstance(payload, dict):
        return records

    if {"dataset", "shot", "seed", "method"}.issubset(payload):
        record = dict(payload)
        record["source_file"] = str(source_file)
        records.append(record)
        return records

    for key in ("results", "records", "runs", "benchmark", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                records.extend(extract_records(item, source_file))
    return records


def read_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in iter_candidate_json_files(results_dir):
        payload = load_json(path)
        if payload is None:
            continue
        records.extend(extract_records(payload, path))
    return records


def record_key(record: Dict[str, Any]) -> Optional[Tuple[str, Any, Any]]:
    dataset = record.get("dataset")
    shot = record.get("shot")
    seed = record.get("seed")
    if dataset is None or shot is None or seed is None:
        return None
    return str(dataset), safe_int(shot), safe_int(seed)


def get_per_class_accuracy(record: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not record:
        return {}
    value = record.get("per_class_accuracy")
    if not isinstance(value, dict):
        return {}
    return {str(label): safe_float(acc) for label, acc in value.items()}


def compute_per_class_delta(method_record: Optional[Dict[str, Any]], vanilla_record: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    method_acc = get_per_class_accuracy(method_record)
    vanilla_acc = get_per_class_accuracy(vanilla_record)
    labels = sorted(set(method_acc) | set(vanilla_acc), key=str)
    delta: Dict[str, Optional[float]] = {}
    for label in labels:
        left = method_acc.get(label)
        right = vanilla_acc.get(label)
        delta[label] = None if left is None or right is None else left - right
    return delta


def group_records(records: List[Dict[str, Any]]) -> Dict[Tuple[str, Any, Any], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = record_key(record)
        method = normalize_method(record.get("method"))
        if key is None or method is None:
            continue
        grouped[key][method] = record
    return grouped


def build_raw_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped = group_records(records)
    for (dataset, shot, seed), methods in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        vanilla = methods.get("vanilla")
        vanilla_accuracy = safe_float(vanilla.get("accuracy")) if vanilla else None
        vanilla_macro_f1 = safe_float(vanilla.get("macro_f1")) if vanilla else None
        method_accuracy = {
            method: safe_float(methods[method].get("accuracy")) if method in methods else None for method in METHODS
        }

        for method in METHODS:
            record = methods.get(method)
            if record is None:
                continue
            accuracy = safe_float(record.get("accuracy"))
            macro_f1 = safe_float(record.get("macro_f1"))
            accuracy_delta = None if accuracy is None or vanilla_accuracy is None else accuracy - vanilla_accuracy
            macro_f1_delta = None if macro_f1 is None or vanilla_macro_f1 is None else macro_f1 - vanilla_macro_f1
            per_class_delta = compute_per_class_delta(record, vanilla)
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "seed": seed,
                    "method": method,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "vanilla_accuracy": method_accuracy["vanilla"],
                    "random_aug_accuracy": method_accuracy["random_aug"],
                    "error_driven_accuracy": method_accuracy["error_driven"],
                    "accuracy_delta": accuracy_delta,
                    "macro_f1_delta": macro_f1_delta,
                    "per_class_accuracy_delta": json.dumps(per_class_delta, ensure_ascii=False, sort_keys=True),
                    "source_file": record.get("source_file", ""),
                }
            )
    return rows


def numeric_values(rows: List[Dict[str, Any]], field: str) -> List[float]:
    values = [safe_float(row.get(field)) for row in rows]
    return [value for value in values if value is not None]


def mean_or_blank(values: List[float]) -> Any:
    return mean(values) if values else ""


def std_or_blank(values: List[float]) -> Any:
    return stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def build_summary_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Any, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["dataset"], row["shot"], row["method"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, shot, method), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        accuracies = numeric_values(rows, "accuracy")
        macro_f1s = numeric_values(rows, "macro_f1")
        deltas = numeric_values(rows, "accuracy_delta")
        summary_rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "method": method,
                "mean_accuracy": mean_or_blank(accuracies),
                "std_accuracy": std_or_blank(accuracies),
                "mean_macro_f1": mean_or_blank(macro_f1s),
                "std_macro_f1": std_or_blank(macro_f1s),
                "mean_improvement_delta": mean_or_blank(deltas),
                "std_improvement_delta": std_or_blank(deltas),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def print_compact_table(summary_rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[(row["dataset"], row["shot"])][row["method"]] = row

    headers = ["dataset", "shot", "vanilla", "random_aug", "error_driven", "error_driven_delta"]
    table_rows = []
    for (dataset, shot), methods in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        table_rows.append(
            [
                str(dataset),
                str(shot),
                fmt(methods.get("vanilla", {}).get("mean_accuracy")),
                fmt(methods.get("random_aug", {}).get("mean_accuracy")),
                fmt(methods.get("error_driven", {}).get("mean_accuracy")),
                fmt(methods.get("error_driven", {}).get("mean_improvement_delta")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(cells: List[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(line(row))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    records = read_records(results_dir)
    raw_rows = build_raw_rows(records)
    summary_rows = build_summary_rows(raw_rows)

    write_csv(output_dir / "raw_results.csv", raw_rows, RAW_FIELDS)
    write_csv(output_dir / "summary_results.csv", summary_rows, SUMMARY_FIELDS)

    print(f"Read {len(records)} result records from {results_dir}")
    print(f"Wrote {output_dir / 'raw_results.csv'}")
    print(f"Wrote {output_dir / 'summary_results.csv'}")
    if summary_rows:
        print()
        print_compact_table(summary_rows)
    else:
        print("No usable benchmark records found.")


if __name__ == "__main__":
    main()
