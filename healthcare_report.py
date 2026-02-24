"""
ParaDetect AI - Healthcare support: batch diagnosis and summary report.
Generates a simple report for healthcare professionals from a folder of smear images.
"""

import os
import argparse
from collections import Counter
from datetime import datetime

from predict import load_model, predict_folder
from config import IMG_SIZE, CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="ParaDetect AI - Batch diagnosis report")
    parser.add_argument("--model", "-m", required=True, help="Path to .keras model")
    parser.add_argument("--folder", "-f", required=True, help="Folder containing smear images")
    parser.add_argument("--output", "-o", default=None, help="Output report file (default: print)")
    args = parser.parse_args()

    model = load_model(args.model)
    results = list(predict_folder(model, args.folder))

    # Count diagnoses
    valid = [(path, label, conf) for path, label, conf in results if label is not None]
    errors = [path for path, label, conf in results if label is None]

    counter = Counter(label for _, label, _ in valid)
    n_total = len(valid)
    n_parasitized = counter.get("Parasitized", 0)
    n_uninfected = counter.get("Uninfected", 0)

    lines = [
        "=" * 60,
        "ParaDetect AI - Malaria Diagnosis Report",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Folder: {os.path.abspath(args.folder)}",
        "",
        "SUMMARY",
        "-" * 40,
        f"Total images analyzed: {n_total}",
        f"Parasitized (positive): {n_parasitized} ({100*n_parasitized/max(1,n_total):.1f}%)",
        f"Uninfected (negative):  {n_uninfected} ({100*n_uninfected/max(1,n_total):.1f}%)",
        f"Errors / unreadable:    {len(errors)}",
        "",
        "NOTE: This is an automated screening aid. Final diagnosis must be confirmed by a qualified professional.",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"Report saved to {args.output}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
