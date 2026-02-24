"""
ParaDetect AI - Training script for malaria diagnosis model.
Run from project root: python train.py
"""

import os
import argparse
from datetime import datetime

from tensorflow import keras

from config import (
    DATA_DIR,
    MODEL_SAVE_DIR,
    RESULTS_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    RANDOM_SEED,
    DROPOUT_RATE,
    NUM_CLASSES,
)
from src.data_loader import get_dataset_from_folders, get_class_weights
from src.model import build_cnn


def main():
    parser = argparse.ArgumentParser(description="Train ParaDetect AI malaria model")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to cell_images folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--save-name", default=None, help="Model save name (default: paradetect_YYYYMMDD_HHMM)")
    args = parser.parse_args()

    print("Loading dataset...")
    train_ds, val_ds = get_dataset_from_folders(
        data_dir=args.data_dir,
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
        validation_split=VALIDATION_SPLIT,
        seed=RANDOM_SEED,
    )

    class_weights = get_class_weights(args.data_dir)
    if class_weights:
        print("Class weights:", class_weights)

    print("Building model...")
    model = build_cnn(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    )
    model.summary()

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_name = args.save_name or f"paradetect_{timestamp}"
    model_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Save training history for analysis
    import json
    history_path = os.path.join(RESULTS_DIR, f"history_{save_name}.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.history.items()},
            f,
            indent=2,
        )
    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")
    return model, history


if __name__ == "__main__":
    main()
