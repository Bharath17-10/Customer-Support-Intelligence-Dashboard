"""Optional sentence-embedding experiment using the same data splits as the baseline."""

import argparse
import json
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocess import load_dataset_with_report
from train import split_indices

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def _evaluate(classifier, embeddings, labels):
    predictions = classifier.predict(embeddings)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=classifier.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classifier.classes_
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "tickets.csv"))
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame, _ = load_dataset_with_report(args.data)
    encoder = SentenceTransformer(args.encoder)
    texts = frame["ticket_text"].tolist()
    embeddings = encoder.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    results = {}
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    for task in ("department", "urgency"):
        labels = frame[task]
        train_indices, validation_indices, test_indices, stratified_splits = (
            split_indices(labels, random_state=args.seed)
        )
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(embeddings[train_indices], labels.iloc[train_indices])
        validation = _evaluate(
            classifier,
            embeddings[validation_indices],
            labels.iloc[validation_indices],
        )
        test = _evaluate(
            classifier,
            embeddings[test_indices],
            labels.iloc[test_indices],
        )
        results[task] = {
            "split_counts": {
                "train": int(len(train_indices)),
                "validation": int(len(validation_indices)),
                "test": int(len(test_indices)),
            },
            "stratified_splits": stratified_splits,
            "validation": validation,
            "test": test,
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "labels": [str(label) for label in classifier.classes_],
        }
        joblib.dump(
            {"encoder_name": args.encoder, "classifier": classifier},
            MODEL_DIR / f"{task}_embeddings.joblib",
        )

    with (REPORT_DIR / "embedding_evaluation.json").open(
        "w", encoding="utf-8"
    ) as report_file:
        json.dump(results, report_file, indent=2)
    print("Embedding experiment complete. Results saved to reports/embedding_evaluation.json")


if __name__ == "__main__":
    main()
