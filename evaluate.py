"""
ParaDetect AI - Model accuracy analysis and evaluation.
Computes accuracy, precision, recall, F1, AUC, and confusion matrix.
"""

import os
import argparse
import numpy as np
import json
from datetime import datetime

from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR, IMG_SIZE, CLASS_NAMES
from src.data_loader import get_dataset_from_folders


def get_predictions_and_labels(model, dataset):
    """Run model on dataset and return flat arrays of predictions and labels."""
    y_true, y_pred = [], []
    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ParaDetect AI model")
    parser.add_argument("--model", "-m", required=True, help="Path to .keras model")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to cell_images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=RESULTS_DIR, help="Where to save plots/report")
    args = parser.parse_args()

    print("Loading model...")
    model = keras.models.load_model(args.model)

    print("Loading validation data...")
    _, val_ds = get_dataset_from_folders(
        data_dir=args.data_dir,
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
    )

    print("Running evaluation...")
    y_true, y_pred = get_predictions_and_labels(model, val_ds)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    report = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }

    print("\n" + "=" * 50)
    print("ParaDetect AI - Model Accuracy Analysis")
    print("=" * 50)
    print(f"Accuracy:    {acc:.2%}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix:")
    print(cm)

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax)
    plt.title("ParaDetect AI - Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")


if __name__ == "__main__":
    main()
