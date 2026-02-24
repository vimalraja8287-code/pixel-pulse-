"""Data loading and dataset preparation for malaria cell images."""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED, CLASS_NAMES


def get_dataset_from_folders(
    data_dir: str = DATA_DIR,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
    subset: str = None,
):
    """
    Load train/validation datasets from folder structure:
      data_dir/Parasitized/*.png
      data_dir/Uninfected/*.png
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please download the malaria cell images dataset and extract to:\n"
            "  paradetect_ai/data/cell_images/Parasitized/\n"
            "  paradetect_ai/data/cell_images/Uninfected/"
        )

    train_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        shuffle=True,
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        shuffle=True,
    )

    # Normalize to [0,1] in pipeline
    normalization = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    return train_ds, val_ds


def get_class_weights(data_dir: str = DATA_DIR):
    """Compute class weights for imbalanced data (optional)."""
    counts = []
    for c in CLASS_NAMES:
        path = os.path.join(data_dir, c)
        if os.path.isdir(path):
            counts.append(len([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]))
        else:
            counts.append(0)
    total = sum(counts)
    if total == 0:
        return None
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}
    return weights
