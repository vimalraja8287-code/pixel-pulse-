"""Configuration for ParaDetect AI - Malaria diagnosis model."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cell_images")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset: expect structure data/cell_images/Parasitized/ and Uninfected/
CLASS_NAMES = ["Uninfected", "Parasitized"]
NUM_CLASSES = 2

# Image
IMG_SIZE = (128, 128)
IMG_SHAPE = (*IMG_SIZE, 3)

# Training
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model
DROPOUT_RATE = 0.4

for d in [DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
