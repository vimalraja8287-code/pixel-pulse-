"""
ParaDetect AI - Inference: automated malaria diagnosis from blood smear images.
Run: python predict.py --model path/to/model.keras --image path/to/cell.png
     python predict.py --model path/to/model.keras --folder path/to/cells/
"""

import os
import argparse
import numpy as np
from tensorflow import keras

from config import MODEL_SAVE_DIR, IMG_SIZE, CLASS_NAMES
from src.preprocess import load_and_preprocess


def load_model(path: str):
    return keras.models.load_model(path)


def predict_single_image(model, image_path: str):
    """Run diagnosis on a single image. Returns class index, label, and confidence."""
    img = load_and_preprocess(image_path, target_size=IMG_SIZE)
    img_batch = np.expand_dims(img, axis=0)
    probs = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


def predict_folder(model, folder_path: str, extensions=(".png", ".jpg", ".jpeg")):
    """Run diagnosis on all images in a folder. Yields (filepath, label, confidence)."""
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(extensions):
            path = os.path.join(folder_path, f)
            try:
                idx, label, conf, _ = predict_single_image(model, path)
                yield path, label, conf
            except Exception as e:
                yield path, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="ParaDetect AI - Malaria diagnosis")
    parser.add_argument("--model", "-m", required=True, help="Path to saved .keras model")
    parser.add_argument("--image", "-i", help="Single image path")
    parser.add_argument("--folder", "-f", help="Folder of images to diagnose")
    parser.add_argument("--output", "-o", help="Output CSV path for batch results")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    print("Loading model...")
    model = load_model(args.model)

    if args.image:
        idx, label, conf, probs = predict_single_image(model, args.image)
        print(f"Image: {args.image}")
        print(f"Diagnosis: {label} (confidence: {conf:.2%})")
        for name, p in zip(CLASS_NAMES, probs):
            print(f"  {name}: {p:.2%}")
        return

    # Batch from folder
    rows = [["file", "diagnosis", "confidence"]]
    for path, label, conf in predict_folder(model, args.folder):
        if label is None:
            rows.append([path, "ERROR", str(conf)])
        else:
            rows.append([path, label, f"{conf:.4f}"])

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"Results written to {args.output}")
    else:
        for row in rows:
            print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
