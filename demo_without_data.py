"""
Quick demo: verify installation and model build without downloading the full dataset.
Run from paradetect_ai: python demo_without_data.py
"""

import os
import sys

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tensorflow import keras

from config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE
from src.model import build_cnn
from src.preprocess import load_and_preprocess

def main():
    print("ParaDetect AI - Demo (no dataset required)")
    print("-" * 40)

    # 1. Build model
    print("1. Building CNN...")
    model = build_cnn(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    )
    print("   OK - Model built.")

    # 2. Dummy forward pass
    print("2. Running dummy forward pass...")
    dummy = np.random.rand(2, *IMG_SIZE, 3).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, NUM_CLASSES)
    print("   OK - Output shape:", out.shape)

    # 3. Preprocess (if we had a real image we'd use load_and_preprocess path)
    print("3. Preprocessing check (synthetic)...")
    import tempfile
    import cv2
    synthetic = np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
    synthetic_bgr = cv2.cvtColor(synthetic, cv2.COLOR_RGB2BGR)
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)  # Windows: close handle so file can be deleted later
    try:
        cv2.imwrite(tmp_path, synthetic_bgr)
        img = load_and_preprocess(tmp_path, target_size=IMG_SIZE)
        assert img.shape == (*IMG_SIZE, 3) and img.dtype == np.float32
        print("   OK - Preprocess output shape:", img.shape)
    finally:
        try:
            os.remove(tmp_path)
        except PermissionError:
            # If antivirus/indexer holds the file briefly, retry once.
            import time
            time.sleep(0.2)
            os.remove(tmp_path)

    print("-" * 40)
    print("Demo complete. Next steps:")
    print("  1. Download malaria cell images to data/cell_images/Parasitized and Uninfected/")
    print("  2. Run: python train.py")
    print("  3. Run: python predict.py -m models/paradetect_*.keras -i <image>")
    print("  4. Run: python evaluate.py -m models/paradetect_*.keras")


if __name__ == "__main__":
    main()
