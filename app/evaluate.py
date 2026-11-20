"""Evaluation suite for legal clause classification with baseline comparisons and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.classifier import DEFAULT_CLASSES, LegalClauseClassifier, get_classifier

DEFAULT_DATASET = Path("data/clause_dataset.json")


def load_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Load the labeled clause dataset."""
    filepath = Path(path or DEFAULT_DATASET)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return json.loads(filepath.read_text(encoding="utf-8"))


def evaluate_classifier(
    classifier: LegalClauseClassifier | None = None,
    dataset_path: str | Path | None = None,
    split: str = "test",
    fast_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate classification accuracy, precision, recall, and F1 on example legal clauses.
    
    Compares performance against random-chance and majority-class baselines.
    """
    if classifier is None:
        classifier = get_classifier(fast_fallback=fast_fallback)

    data = load_dataset(dataset_path)
    samples = data.get(split, [])
    classes = data.get("classes", DEFAULT_CLASSES)

    if not samples:
        raise ValueError(f"No samples found in split '{split}'")

    y_true = [s["label"] for s in samples]
    y_pred = []
    sample_results = []

    for item in samples:
        res = classifier.predict(item["text"])
        pred_label = res["predicted_label"]
        y_pred.append(pred_label)
        sample_results.append({
            "text": item["text"],
            "true_label": item["label"],
            "predicted_label": pred_label,
            "confidence": res["confidence"],
            "correct": pred_label == item["label"],
        })

    # Total and accuracy
    total = len(y_true)
    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_count / total if total > 0 else 0.0

    # Baselines
    baseline_random = 1.0 / len(classes) if classes else 0.2
    from collections import Counter
    class_counts = Counter(y_true)
    most_common_count = class_counts.most_common(1)[0][1] if class_counts else 0
    baseline_majority = most_common_count / total if total > 0 else 0.2

    # Per-class metrics: precision, recall, f1
    per_class_metrics: dict[str, dict[str, float | int]] = {}
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        support = sum(1 for yt in y_true if yt == cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / max(len(classes), 1)

    # Confusion matrix
    confusion_matrix: dict[str, dict[str, int]] = {
        actual: {pred: 0 for pred in classes} for actual in classes
    }
    for yt, yp in zip(y_true, y_pred):
        if yt in confusion_matrix and yp in confusion_matrix[yt]:
            confusion_matrix[yt][yp] += 1

    return {
        "split": split,
        "total_samples": total,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "baseline_random_accuracy": round(baseline_random, 4),
        "baseline_majority_accuracy": round(baseline_majority, 4),
        "accuracy_gain_vs_random": round(accuracy - baseline_random, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "sample_results": sample_results,
        "limits": [
            "Model is specialized for the 5 trained commercial clause categories and will classify non-legal text into the nearest category.",
            "Short fragments without standard keywords or legal context rely on prior distribution.",
        ],
    }
