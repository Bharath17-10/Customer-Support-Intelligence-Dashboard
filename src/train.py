import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import load_dataset_with_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def _can_stratify(labels, test_size):
    class_counts = labels.value_counts()
    test_count = math.ceil(len(labels) * test_size)
    train_count = len(labels) - test_count
    return (
        len(class_counts) > 1
        and int(class_counts.min()) >= 2
        and test_count >= len(class_counts)
        and train_count >= len(class_counts)
    )


def split_indices(labels, random_state=42):
    indices = np.arange(len(labels))
    test_stratified = _can_stratify(labels, 0.2)
    train_validation, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=labels if test_stratified else None,
    )

    train_validation_labels = labels.iloc[train_validation]
    validation_stratified = _can_stratify(train_validation_labels, 0.25)
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=random_state,
        stratify=train_validation_labels if validation_stratified else None,
    )
    train_labels = set(labels.iloc[train].unique())
    all_labels = set(labels.unique())
    if train_labels != all_labels:
        missing_labels = sorted(all_labels - train_labels)
        raise ValueError(
            "The reproducible training split would omit labels "
            f"{missing_labels} from the training set. Add examples for rare labels."
        )
    return train, validation, test, {
        "test_stratified": test_stratified,
        "validation_stratified": validation_stratified,
    }


def _evaluate(model, texts, labels):
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=model.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=model.classes_
        ).tolist(),
    }, predictions


def _class_counts(labels):
    return {
        str(label): int(count)
        for label, count in labels.value_counts().sort_index().items()
    }


def fit_and_evaluate(texts, labels, task_name, random_state=42):
    train_indices, validation_indices, test_indices, stratified_splits = split_indices(
        labels, random_state
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50000,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    model.fit(texts.iloc[train_indices], labels.iloc[train_indices])

    validation_result, _ = _evaluate(
        model, texts.iloc[validation_indices], labels.iloc[validation_indices]
    )
    test_result, test_predictions = _evaluate(
        model, texts.iloc[test_indices], labels.iloc[test_indices]
    )
    result = {
        "split_counts": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "stratified_splits": stratified_splits,
        "validation": validation_result,
        "test": test_result,
        "accuracy": test_result["accuracy"],
        "macro_f1": test_result["macro_f1"],
        "labels": [str(label) for label in model.classes_],
    }
    errors = []
    test_texts = texts.iloc[test_indices]
    test_labels = labels.iloc[test_indices]
    for text, actual, predicted in zip(
        test_texts, test_labels, test_predictions
    ):
        if actual != predicted and len(errors) < 25:
            errors.append(
                {
                    "ticket_text": text,
                    "actual": str(actual),
                    "predicted": str(predicted),
                }
            )
    joblib.dump(model, MODEL_DIR / f"{task_name}_tfidf.joblib")
    return result, errors, train_indices, validation_indices, test_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "tickets.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    frame, dataset_report = load_dataset_with_report(args.data)
    all_results = {}
    all_errors = {}
    split_report = {}

    for task in ("department", "urgency"):
        result, errors, train_indices, validation_indices, test_indices = (
            fit_and_evaluate(
                frame["ticket_text"],
                frame[task],
                task,
                random_state=args.seed,
            )
        )
        all_results[task] = result
        all_errors[task] = errors
        split_report[task] = {
            "train": _class_counts(frame.iloc[train_indices][task]),
            "validation": _class_counts(frame.iloc[validation_indices][task]),
            "test": _class_counts(frame.iloc[test_indices][task]),
        }

    with (REPORT_DIR / "tfidf_evaluation.json").open(
        "w", encoding="utf-8"
    ) as report_file:
        json.dump(all_results, report_file, indent=2)
    with (REPORT_DIR / "error_examples.json").open(
        "w", encoding="utf-8"
    ) as error_file:
        json.dump(all_errors, error_file, indent=2)
    with (REPORT_DIR / "dataset_analysis.json").open(
        "w", encoding="utf-8"
    ) as dataset_file:
        json.dump(
            {
                **dataset_report,
                "random_seed": args.seed,
                "split_distribution": split_report,
            },
            dataset_file,
            indent=2,
        )

    print("Training complete. Models saved in models/ and reports updated.")
    print(
        json.dumps(
            {
                task: {
                    "test_accuracy": result["test"]["accuracy"],
                    "test_macro_f1": result["test"]["macro_f1"],
                }
                for task, result in all_results.items()
            },
            indent=2,
        )
    )
    print(
        "Dataset analysis:",
        json.dumps(
            {
                "input_records": dataset_report["input_records"],
                "usable_records": dataset_report["usable_records"],
                "duplicate_records": dataset_report["duplicate_records"],
                "department_distribution": dataset_report[
                    "department_distribution"
                ],
                "urgency_distribution": dataset_report["urgency_distribution"],
            }
        ),
    )


if __name__ == "__main__":
    main()
