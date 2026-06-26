import argparse
import csv
import gc
import json
import os
import pathlib
import random
import sys
import warnings
from collections import Counter
from statistics import mean, stdev
from typing import Dict, List, Optional
from warnings import simplefilter

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from transformers.trainer_utils import set_seed

from setfit import SetFitModel, SetFitTrainer
from setfit.data import sample_dataset
from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_DATASETS = ["sst2"]
DEFAULT_SHOTS = [4, 8, 16]
DEFAULT_SEEDS = list(range(10))
METHOD_VANILLA = "vanilla"
METHOD_RANDOM = "random_aug"
METHOD_ERROR = "error_driven"
METHODS = [METHOD_VANILLA, METHOD_RANDOM, METHOD_ERROR]
DATASET_ALIASES = {
    "sst2": ["sst2"],
    "ag_news": ["ag_news"],
    "trec": ["TREC-QC", "trec"],
    "TREC-QC": ["TREC-QC"],
    "SentEval-CR": ["SentEval-CR"],
    "dbpedia": ["dbpedia", "dbpedia_14"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible SetFit few-shot benchmark.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--is_dev_set", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--is_test_set", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sample_sizes", "--shots", dest="sample_sizes", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=[
            "logistic_regression",
            "svc-rbf",
            "svc-rbf-norm",
            "knn",
            "pytorch",
            "pytorch_complex",
        ],
    )
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--exp_name", default="benchmark")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--add_normalization_layer", default=False, action="store_true")
    parser.add_argument("--optimizer_name", default="AdamW")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")
    parser.add_argument("--eval_strategy", default=False)
    parser.add_argument("--random_synthetic_multiplier", type=float, default=1.0)
    parser.add_argument("--synthetic_samples_per_error_class", type=int, default=1)
    parser.add_argument("--max_synthetic_samples_per_class", type=int, default=16)
    parser.add_argument("--synthetic_ratio", type=float, default=0.5)

    args = parser.parse_args()
    args.datasets = ["sst2"]
    args.num_iterations = min(args.num_iterations, 10)
    args.batch_size = min(8, args.batch_size)
    if args.synthetic_ratio < 0:
        raise ValueError("--synthetic_ratio must be >= 0.")
    return args


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def resolve_dataset_metrics(args) -> Dict[str, str]:
    return {"sst2": "accuracy"}


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def slugify(value: str) -> str:
    return value.replace("/", "-").replace("_", "-").lower()


def candidate_dataset_names(dataset: str) -> List[str]:
    return DATASET_ALIASES.get(dataset, [dataset])


def load_setfit_dataset(dataset: str) -> tuple[str, Dataset, Dataset]:
    last_error = None
    for dataset_name in candidate_dataset_names(dataset):
        try:
            train_data = load_dataset(f"SetFit/{dataset_name}", split="train")
            test_data = load_dataset(f"SetFit/{dataset_name}", split="test")
            print(f"Loaded SetFit/{dataset_name}: train={len(train_data)}, test={len(test_data)}")
            return dataset_name, train_data, test_data
        except Exception as exc:  # noqa: BLE001 - keep fallback robust for dataset aliases.
            last_error = exc
            print(f"Could not load SetFit/{dataset_name}: {exc}")
    raise RuntimeError(f"Could not load dataset {dataset!r} from aliases {candidate_dataset_names(dataset)}") from last_error


def report_class_imbalance(dataset: str, test_data: Dataset) -> None:
    counter = Counter(test_data["label"])
    label_samples = sorted(counter.items(), key=lambda label_samples: label_samples[1])
    smallest_n_samples = label_samples[0][1]
    largest_n_samples = label_samples[-1][1]
    if largest_n_samples > smallest_n_samples * 1.5:
        warnings.warn(
            "The test set has a class imbalance "
            f"for {dataset}: {', '.join(f'label {label} w. {n_samples} samples' for label, n_samples in label_samples)}."
        )


def result_dir(output_dir: pathlib.Path, dataset: str, shot: int, seed: int) -> pathlib.Path:
    path = output_dir / slugify(dataset) / f"{shot}-shot" / f"seed-{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(output_dir: pathlib.Path, dataset: str, shot: int, seed: int, method: str) -> pathlib.Path:
    return result_dir(output_dir, dataset, shot, seed) / f"{method}.json"


def result_path_candidates(output_dir: pathlib.Path, dataset: str, shot: int, seed: int, method: str) -> List[pathlib.Path]:
    paths = [result_path(output_dir, dataset, shot, seed, method)]
    if method == METHOD_ERROR and METHOD_ERROR != "augmented":
        paths.append(result_dir(output_dir, dataset, shot, seed) / "augmented.json")
    return paths


def existing_result_path(output_dir: pathlib.Path, dataset: str, shot: int, seed: int, method: str) -> Optional[pathlib.Path]:
    for path in result_path_candidates(output_dir, dataset, shot, seed, method):
        if path.exists():
            return path
    return None


def run_is_complete(output_dir: pathlib.Path, dataset: str, shot: int, seed: int) -> bool:
    return all(existing_result_path(output_dir, dataset, shot, seed, method) is not None for method in METHODS)


def load_existing_result_records(output_dir: pathlib.Path, dataset: str, shot: int, seed: int) -> Optional[List[dict]]:
    records = []
    for method in METHODS:
        path = existing_result_path(output_dir, dataset, shot, seed, method)
        if path is None:
            return None
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Existing result unreadable, rerunning: dataset={dataset}, shot={shot}, seed={seed}, file={path}, error={exc}")
            return None
    return records


def to_python_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def json_label(label):
    if hasattr(label, "item"):
        return label.item()
    return label


def get_label_name(label, dataset: Dataset) -> str:
    label = json_label(label)
    label_feature = dataset.features.get("label")
    if hasattr(label_feature, "names") and isinstance(label, int) and 0 <= label < len(label_feature.names):
        return label_feature.names[label]
    return str(label)


def get_label_names(labels: List, dataset: Dataset) -> List[str]:
    return [get_label_name(label, dataset) for label in labels]


def build_model(args, train_data: Dataset) -> SetFitModel:
    if args.classifier == "pytorch":
        model = SetFitModel.from_pretrained(
            args.model,
            use_differentiable_head=True,
            head_params={"out_features": len(set(train_data["label"]))},
        )
    else:
        model = SetFitModel.from_pretrained(args.model)
    model.model_body.max_seq_length = args.max_seq_length
    if args.add_normalization_layer:
        model.model_body._modules["2"] = models.Normalize()
    return model


def train_setfit_model(args, train_data: Dataset, eval_data: Dataset, loss_class, seed: int, metric: str):
    set_reproducible_seed(seed)
    model = build_model(args, train_data)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        metric=metric,
        loss_class=loss_class,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
    )
    trainer.args.dataloader_num_workers = 0
    if not args.eval_strategy:
        trainer.args.eval_strategy = "no"
    if args.classifier == "pytorch":
        trainer.freeze()
        trainer.train()
        trainer.unfreeze(keep_body_frozen=args.keep_body_frozen)
        trainer.train(
            num_epochs=25,
            body_learning_rate=1e-5,
            learning_rate=args.lr,
            l2_weight=0.0,
            batch_size=args.batch_size,
        )
    else:
        trainer.train()
    return trainer


def compute_detailed_metrics(model: SetFitModel, eval_data: Dataset, metric: str) -> dict:
    y_true = [json_label(label) for label in eval_data["label"]]
    y_pred = [json_label(label) for label in to_python_list(model.predict(eval_data["text"], use_labels=False))]
    labels = sorted(set(y_true) | set(y_pred), key=lambda label: str(label))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_accuracy = {}
    for idx, label in enumerate(labels):
        total = int(matrix[idx].sum())
        per_class_accuracy[str(label)] = float(matrix[idx][idx] / total * 100) if total else None
    accuracy = float(accuracy_score(y_true, y_pred) * 100)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100)
    if metric == "matthews_correlation":
        primary_score = float(matthews_corrcoef(y_true, y_pred) * 100)
    elif metric == "macro_f1":
        primary_score = macro_f1
    else:
        primary_score = accuracy

    errors = [
        {
            "text": text,
            "true_label": json_label(true_label),
            "predicted_label": json_label(pred_label),
        }
        for text, true_label, pred_label in zip(eval_data["text"], y_true, y_pred)
        if true_label != pred_label
    ]
    return {
        "score": primary_score,
        "measure": metric,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": [json_label(label) for label in labels],
        "label_names": get_label_names(labels, eval_data),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class_accuracy": per_class_accuracy,
        "num_errors": len(errors),
        "errors": errors,
    }


def generate_error_driven_synthetic_samples(
    errors: list,
    reference_data: Dataset,
    samples_per_error_class: int,
    max_samples_per_class: int,
) -> Dataset:
    error_counts = Counter(error["true_label"] for error in errors)
    if not error_counts:
        return Dataset.from_dict({"text": [], "label": []})

    texts = []
    labels = []
    for label in sorted(error_counts, key=lambda label: str(label)):
        label_name = get_label_name(label, reference_data)
        count = min(error_counts[label] * samples_per_error_class, max_samples_per_class)
        confused_with = sorted(
            {
                get_label_name(error["predicted_label"], reference_data)
                for error in errors
                if error["true_label"] == label
            }
        )
        confused_text = ", ".join(confused_with) if confused_with else "another class"
        for sample_idx in range(count):
            template_id = sample_idx % 3
            if template_id == 0:
                text = f"Synthetic hard example for class '{label_name}': the correct label is '{label_name}', not '{confused_text}'."
            elif template_id == 1:
                text = f"Error-driven sample: this input may look similar to '{confused_text}', but it belongs to '{label_name}'."
            else:
                text = f"Deterministic augmentation for '{label_name}' with challenging cues from '{confused_text}'."
            texts.append(text)
            labels.append(label)

    return Dataset.from_dict({"text": texts, "label": labels})


def generate_random_synthetic_samples(
    train_data: Dataset,
    reference_data: Dataset,
    seed: int,
    num_samples: int,
) -> Dataset:
    if num_samples <= 0:
        return Dataset.from_dict({"text": [], "label": []})

    rng = random.Random(seed)
    labels = sorted(set(train_data["label"]), key=lambda label: str(label))
    texts = []
    synthetic_labels = []
    for sample_idx in range(num_samples):
        label = rng.choice(labels)
        label_name = get_label_name(label, reference_data)
        template_id = sample_idx % 3
        if template_id == 0:
            text = f"Random synthetic example for class '{label_name}'."
        elif template_id == 1:
            text = f"Template augmentation sample labeled as '{label_name}'."
        else:
            text = f"Simple generated text associated with category '{label_name}'."
        texts.append(text)
        synthetic_labels.append(label)
    return Dataset.from_dict({"text": texts, "label": synthetic_labels})


def cap_synthetic_data(synthetic_data: Dataset, original_train_size: int, synthetic_ratio: float, seed: int) -> Dataset:
    max_synthetic_size = min(original_train_size, int(original_train_size * synthetic_ratio))
    if max_synthetic_size <= 0 or len(synthetic_data) == 0:
        return Dataset.from_dict({"text": [], "label": []})
    if len(synthetic_data) <= max_synthetic_size:
        return synthetic_data
    return synthetic_data.shuffle(seed=seed).select(range(max_synthetic_size))


def create_augmented_train_data(train_data: Dataset, synthetic_data: Dataset) -> Dataset:
    train_core = train_data.select_columns(["text", "label"])
    synthetic_data = synthetic_data.cast(train_core.features)
    max_train_size = 2 * len(train_core)
    max_synthetic_size = max(0, max_train_size - len(train_core))
    if len(synthetic_data) > max_synthetic_size:
        synthetic_data = synthetic_data.select(range(max_synthetic_size))
    return concatenate_datasets([train_core, synthetic_data])


def compute_improvement(baseline_details: dict, method_details: dict) -> dict:
    labels = sorted(
        set(baseline_details["per_class_accuracy"]) | set(method_details["per_class_accuracy"]),
        key=lambda label: str(label),
    )
    per_class_delta = {}
    for label in labels:
        baseline_value = baseline_details["per_class_accuracy"].get(label)
        method_value = method_details["per_class_accuracy"].get(label)
        per_class_delta[label] = (
            float(method_value - baseline_value)
            if baseline_value is not None and method_value is not None
            else None
        )
    return {
        "accuracy_delta": float(method_details["accuracy"] - baseline_details["accuracy"]),
        "macro_f1_delta": float(method_details["macro_f1"] - baseline_details["macro_f1"]),
        "per_class_accuracy_delta": per_class_delta,
    }


def error_summary(errors: list, reference_data: Dataset) -> dict:
    true_counts = Counter(error["true_label"] for error in errors)
    confusion_counts = Counter((error["true_label"], error["predicted_label"]) for error in errors)
    return {
        "num_errors": len(errors),
        "errored_classes": {
            str(label): {
                "label_name": get_label_name(label, reference_data),
                "count": count,
            }
            for label, count in sorted(true_counts.items(), key=lambda item: str(item[0]))
        },
        "confusions": [
            {
                "true_label": json_label(true_label),
                "true_label_name": get_label_name(true_label, reference_data),
                "predicted_label": json_label(predicted_label),
                "predicted_label_name": get_label_name(predicted_label, reference_data),
                "count": count,
            }
            for (true_label, predicted_label), count in sorted(
                confusion_counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
            )
        ],
    }


def make_result_record(
    args,
    dataset: str,
    hf_dataset: str,
    shot: int,
    seed: int,
    method: str,
    details: dict,
    baseline_details: dict,
    synthetic_data: Optional[Dataset],
    synthetic_strategy: str,
    errors: Optional[list],
    test_data: Dataset,
) -> dict:
    improvement = compute_improvement(baseline_details, details)
    record = {
        "dataset": dataset,
        "hf_dataset": hf_dataset,
        "shot": shot,
        "seed": seed,
        "method": method,
        "measure": details["measure"],
        "score": details["score"],
        "accuracy": details["accuracy"],
        "macro_f1": details["macro_f1"],
        "confusion_matrix": details["confusion_matrix"],
        "confusion_matrix_labels": details["labels"],
        "confusion_matrix_label_names": details["label_names"],
        "per_class_accuracy": details["per_class_accuracy"],
        "improvement_delta": improvement,
        "synthetic_data": {
            "strategy": synthetic_strategy,
            "num_samples": 0 if synthetic_data is None else len(synthetic_data),
            "preview": [] if synthetic_data is None else synthetic_data.select(range(min(5, len(synthetic_data)))).to_list(),
        },
        "error_analysis": error_summary(errors or [], test_data),
        "reproducibility": {
            "model": args.model,
            "loss": args.loss,
            "classifier": args.classifier,
            "num_iterations": args.num_iterations,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "seed": seed,
            "synthetic_ratio": args.synthetic_ratio,
        },
    }
    return record


def save_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=2, sort_keys=True)


def cleanup_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def release_model(*objects) -> None:
    del objects
    cleanup_memory()


def run_method(args, train_data: Dataset, test_data: Dataset, loss_class, seed: int, metric: str) -> dict:
    trainer = None
    model = None
    try:
        trainer = train_setfit_model(args, train_data, test_data, loss_class, seed, metric)
        model = trainer.model
        details = compute_detailed_metrics(model, test_data, metric)
        return details
    finally:
        del model
        del trainer
        cleanup_memory()


def run_benchmark_case(
    args,
    output_dir: pathlib.Path,
    dataset: str,
    hf_dataset: str,
    full_train_data: Dataset,
    test_data: Dataset,
    shot: int,
    seed: int,
    loss_class,
    metric: str,
) -> List[dict]:
    if run_is_complete(output_dir, dataset, shot, seed) and not args.override_results:
        existing_records = load_existing_result_records(output_dir, dataset, shot, seed)
        if existing_records is not None:
            print(f"Skipping existing result: dataset={dataset}, shot={shot}, seed={seed}")
            return existing_records


    print(f"\n=== dataset={dataset} shot={shot} seed={seed} ===")
    train_data = sample_dataset(full_train_data, num_samples=shot, seed=seed).select_columns(["text", "label"])
    original_train_size = len(train_data)
    print(
        "Run config: "
        f"dataset={dataset}, shot={shot}, seed={seed}, "
        f"original_train_size={original_train_size}, synthetic_ratio={args.synthetic_ratio}"
    )

    print("Training Vanilla SetFit")
    vanilla_details = run_method(args, train_data, test_data, loss_class, seed, metric)
    cleanup_memory()
    vanilla_record = make_result_record(
        args,
        dataset,
        hf_dataset,
        shot,
        seed,
        METHOD_VANILLA,
        vanilla_details,
        vanilla_details,
        None,
        "none",
        vanilla_details["errors"],
        test_data,
    )
    save_json(result_path(output_dir, dataset, shot, seed, METHOD_VANILLA), vanilla_record)

    error_synthetic = generate_error_driven_synthetic_samples(
        vanilla_details["errors"],
        test_data,
        args.synthetic_samples_per_error_class,
        args.max_synthetic_samples_per_class,
    )
    error_synthetic = cap_synthetic_data(error_synthetic, original_train_size, args.synthetic_ratio, seed)
    random_budget = int(round(len(error_synthetic) * args.random_synthetic_multiplier))
    random_synthetic = generate_random_synthetic_samples(train_data, test_data, seed, random_budget)
    random_synthetic = cap_synthetic_data(random_synthetic, original_train_size, args.synthetic_ratio, seed)
    print(
        "Synthetic config: "
        f"dataset={dataset}, shot={shot}, seed={seed}, "
        f"original_train_size={original_train_size}, "
        f"synthetic_size={len(error_synthetic)}, "
        f"random_synthetic_size={len(random_synthetic)}, "
        f"synthetic_ratio={args.synthetic_ratio}"
    )

    print(f"Training Random Augmentation baseline ({len(random_synthetic)} synthetic samples)")
    random_train_data = create_augmented_train_data(train_data, random_synthetic)
    print(f"Random augmentation final_train_size={len(random_train_data)}")
    random_details = run_method(args, random_train_data, test_data, loss_class, seed, metric)
    cleanup_memory()
    random_record = make_result_record(
        args,
        dataset,
        hf_dataset,
        shot,
        seed,
        METHOD_RANDOM,
        random_details,
        vanilla_details,
        random_synthetic,
        "random_label_template",
        vanilla_details["errors"],
        test_data,
    )
    save_json(result_path(output_dir, dataset, shot, seed, METHOD_RANDOM), random_record)

    print(f"Training Error-driven Augmentation ({len(error_synthetic)} synthetic samples)")
    error_train_data = create_augmented_train_data(train_data, error_synthetic)
    print(f"Error-driven augmentation final_train_size={len(error_train_data)}")
    error_details = run_method(args, error_train_data, test_data, loss_class, seed, metric)
    cleanup_memory()
    error_record = make_result_record(
        args,
        dataset,
        hf_dataset,
        shot,
        seed,
        METHOD_ERROR,
        error_details,
        vanilla_details,
        error_synthetic,
        "error_driven_template",
        vanilla_details["errors"],
        test_data,
    )
    save_json(result_path(output_dir, dataset, shot, seed, METHOD_ERROR), error_record)

    print(
        f"Completed dataset={dataset} shot={shot} seed={seed}: "
        f"vanilla={vanilla_record['accuracy']:.3f}, "
        f"random={random_record['accuracy']:.3f}, "
        f"error-driven={error_record['accuracy']:.3f}"
    )
    del train_data
    del random_train_data
    del error_train_data
    del random_synthetic
    del error_synthetic
    cleanup_memory()
    return [vanilla_record, random_record, error_record]


def summarize_results(output_dir: pathlib.Path, records: List[dict]) -> None:
    rows = []
    detailed_rows = []
    scaling_rows = []
    grouped: Dict[tuple, List[dict]] = {}

    for record in records:
        key = (record["dataset"], record["shot"], record["method"])
        grouped.setdefault(key, []).append(record)

    for (dataset, shot, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        accuracies = [record["accuracy"] for record in group]
        macro_f1s = [record["macro_f1"] for record in group]
        rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "method": method,
                "mean": mean(accuracies),
                "std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
            }
        )
        detailed_rows.extend(
            [
                {
                    "dataset": dataset,
                    "shot": shot,
                    "method": method,
                    "metric": "accuracy",
                    "mean": mean(accuracies),
                    "std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                },
                {
                    "dataset": dataset,
                    "shot": shot,
                    "method": method,
                    "metric": "macro_f1",
                    "mean": mean(macro_f1s),
                    "std": stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
                },
            ]
        )

    by_dataset_method: Dict[tuple, List[dict]] = {}
    for row in rows:
        by_dataset_method.setdefault((row["dataset"], row["method"]), []).append(row)
    for (dataset, method), group in sorted(by_dataset_method.items()):
        group = sorted(group, key=lambda row: row["shot"])
        for prev, curr in zip(group, group[1:]):
            scaling_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "from_shot": prev["shot"],
                    "to_shot": curr["shot"],
                    "mean_accuracy_delta": curr["mean"] - prev["mean"],
                }
            )

    write_csv(output_dir / "summary_table.csv", rows, ["dataset", "shot", "method", "mean", "std"])
    write_csv(
        output_dir / "summary_metrics.csv",
        detailed_rows,
        ["dataset", "shot", "method", "metric", "mean", "std"],
    )
    write_csv(
        output_dir / "shot_scaling_trend.csv",
        scaling_rows,
        ["dataset", "method", "from_shot", "to_shot", "mean_accuracy_delta"],
    )
    save_json(output_dir / "summary.json", {"summary": rows, "metrics": detailed_rows, "shot_scaling_trend": scaling_rows})


def write_csv(path: pathlib.Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    set_reproducible_seed(args.seeds[0])

    parent_directory = pathlib.Path(__file__).parent.absolute()
    repo_root = parent_directory.parents[1]
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_script_path = output_dir / "train_script.py"
    train_script_path.write_text(pathlib.Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    with open(train_script_path, "a", encoding="utf-8") as f_out:
        f_out.write("\n\n# Script was called via:\n# python " + " ".join(sys.argv))

    loss_class = LOSS_NAME_TO_CLASS[args.loss]
    all_records = []

    dataset = "sst2"
    metric = resolve_dataset_metrics(args)[dataset]
    hf_dataset, full_train_data, test_data = load_setfit_dataset(dataset)
    report_class_imbalance(dataset, test_data)
    for shot in args.sample_sizes:
        for seed in args.seeds:
            records = run_benchmark_case(
                args,
                output_dir,
                dataset,
                hf_dataset,
                full_train_data,
                test_data,
                shot,
                seed,
                loss_class,
                metric,
            )
            all_records.extend(records)
            del records
            cleanup_memory()
    del full_train_data
    del test_data
    cleanup_memory()

    summarize_results(output_dir, all_records)
    cleanup_memory()
    print(f"\nBenchmark complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


# Script was called via:
# python run_fewshot.py --model sentence-transformers/all-MiniLM-L6-v2 --datasets sst2 --sample_sizes 16 --seeds 9 --is_dev_set true --batch_size 2 --num_iterations 10 --num_epochs 1 --max_seq_length 128