"""
train_model.py — Malaria Detection CNN Training Script
=======================================================
Trains a binary CNN classifier on the Parasitized vs Uninfected malaria
cell-image dataset and saves the result as 'malaria_model.h5'.

Dataset expected structure:
    data/
     └─ cell_images/
           ├─ Parasitized/
           └─ Uninfected/

Run:
    python train_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "cell_images")   # root of Parasitized/ & Uninfected/
MODEL_PATH     = os.path.join(BASE_DIR, "malaria_model.h5")      # where the trained model is saved
HISTORY_PATH   = os.path.join(BASE_DIR, "results", "history_malaria_model.json")

IMG_SIZE       = (64, 64)      # resize every image to 64×64 pixels
BATCH_SIZE     = 32            # images per training step
EPOCHS         = 15            # max training epochs (EarlyStopping may stop sooner)
VALIDATION_SPLIT = 0.20        # 20 % of data held out for validation
SEED           = 42            # reproducibility

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 2. Load & preprocess the dataset
#    keras.utils.image_dataset_from_directory automatically:
#      • reads sub-folder names as class labels
#      • resizes images
#      • splits into train / validation
#    We then rescale pixel values from [0, 255] → [0, 1] via
#    a Rescaling layer inside the model (avoids data leakage).
# ──────────────────────────────────────────────────────────────
print(f"\n📂  Loading images from: {DATA_DIR}")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels       = "inferred",          # use sub-folder names
    label_mode   = "binary",            # 0 or 1 for binary classification
    image_size   = IMG_SIZE,            # resize to 64×64
    batch_size   = BATCH_SIZE,
    validation_split = VALIDATION_SPLIT,
    subset       = "training",
    seed         = SEED,
    shuffle      = True,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels       = "inferred",
    label_mode   = "binary",
    image_size   = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    validation_split = VALIDATION_SPLIT,
    subset       = "validation",
    seed         = SEED,
    shuffle      = False,
)

class_names = train_ds.class_names
print(f"✅  Classes found: {class_names}")
print(f"    Training batches  : {len(train_ds)}")
print(f"    Validation batches: {len(val_ds)}")

# ──────────────────────────────────────────────────────────────
# 3. Performance optimisation — prefetch data while GPU trains
# ──────────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ──────────────────────────────────────────────────────────────
# 4. Build the CNN model
#    Architecture:
#      Rescaling → Conv+Pool (×3) → Flatten → Dense → Dropout → Output
#    • Rescaling(1/255) normalises pixels inside the model graph
#    • Three Conv blocks progressively extract features at
#      increasing abstraction (32 → 64 → 128 filters)
#    • Dropout(0.4) reduces overfitting
#    • Final Dense(1, sigmoid) outputs P(Parasitized)
# ──────────────────────────────────────────────────────────────
print("\n🔧  Building CNN model…")

model = keras.Sequential([
    # -- Normalisation (kept inside model so saved model is self-contained)
    layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 3)),

    # -- Data augmentation (applied only during training)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # -- Convolutional block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # -- Convolutional block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # -- Convolutional block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # -- Classifier head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),   # binary → sigmoid output
], name="malaria_cnn")

model.summary()

# ──────────────────────────────────────────────────────────────
# 5. Compile the model
#    • Adam optimiser with standard learning rate
#    • binary_crossentropy loss (standard for 2-class sigmoid output)
#    • accuracy metric
# ──────────────────────────────────────────────────────────────
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"],
)

# ──────────────────────────────────────────────────────────────
# 6. Callbacks — automatic learning-rate reduction and early stop
# ──────────────────────────────────────────────────────────────
callbacks = [
    # Save the best checkpoint (by val_accuracy)
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1,
    ),
    # Halve LR when val_loss plateaus for 2 epochs
    keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 2,
        min_lr   = 1e-6,
        verbose  = 1,
    ),
    # Stop early if val_loss does not improve for 5 epochs
    keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = 5,
        restore_best_weights = True,
        verbose              = 1,
    ),
]

# ──────────────────────────────────────────────────────────────
# 7. Train the model
# ──────────────────────────────────────────────────────────────
print(f"\n🚀  Training for up to {EPOCHS} epochs…")
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS,
    callbacks       = callbacks,
)

# ──────────────────────────────────────────────────────────────
# 8. Evaluate on validation set
# ──────────────────────────────────────────────────────────────
print("\n📊  Evaluating on validation set…")
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"\n    Validation Loss    : {val_loss:.4f}")
print(f"    Validation Accuracy: {val_acc * 100:.2f}%")

# ──────────────────────────────────────────────────────────────
# 9. Save training history as JSON for later analysis / plotting
# ──────────────────────────────────────────────────────────────
with open(HISTORY_PATH, "w") as f:
    json.dump(
        {k: [float(x) for x in v] for k, v in history.history.items()},
        f,
        indent=2,
    )
print(f"\n💾  Model  saved → {MODEL_PATH}")
print(f"📝  History saved → {HISTORY_PATH}")
print("\n✅  Done!")
