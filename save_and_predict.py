"""
save_and_predict.py
====================
PART 1 — Save a trained Keras model as malaria_model.h5
PART 2 — Load it back and run inference on a single new image

Assumes you already have a trained model in the variable `model`.
If you don't, uncomment the section below that loads the existing model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — Save the trained model to disk
# ──────────────────────────────────────────────────────────────────────────────

# Path where the model will be saved
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malaria_model.h5")

# If you already have a `model` variable in memory (trained in the same session),
# just call model.save().  The .h5 file stores EVERYTHING:
#   • Model architecture (layer types, connections, config)
#   • Trained weights  (learned parameters)
#   • Optimizer state  (so training can be resumed if needed)

# ── Uncomment whichever source applies ────────────────────────────────────────

# Option A: model was trained in this same Python session (most common)
# model.save(MODEL_PATH)

# Option B: load an already-saved .keras model and re-save as .h5
existing_keras = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "paradetect_20260224_1208.keras"
)
model = keras.models.load_model(existing_keras)
model.save(MODEL_PATH)   # converts to .h5 format

print(f"✅ Model saved → {MODEL_PATH}")
print(f"   File size   : {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — Load the saved .h5 and predict on a single new image
# ──────────────────────────────────────────────────────────────────────────────

# Class labels (order must match what was used during training)
CLASS_NAMES = ["Uninfected", "Parasitized"]   # alphabetical = Keras default
IMG_SIZE    = (64, 64)                         # must match training size

# -- Load the saved model from disk (this is all you need for inference) -------
loaded_model = keras.models.load_model(MODEL_PATH)
print(f"\n✅ Model loaded from {MODEL_PATH}")
loaded_model.summary()

# -- Load a single demo image and preprocess it --------------------------------
# Replace this path with any real cell image (.png / .jpg)
demo_image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "cell_images", "Parasitized",
    os.listdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "cell_images", "Parasitized")
    )[0]   # grab the very first image in the folder as a demo
)

# Read → resize → rescale → add batch dimension
img = keras.utils.load_img(demo_image_path, target_size=IMG_SIZE)
img_array = keras.utils.img_to_array(img)          # shape: (64, 64, 3)
img_array = img_array / 255.0                       # rescale to [0, 1]
img_array = np.expand_dims(img_array, axis=0)       # shape: (1, 64, 64, 3)

# -- Run inference -------------------------------------------------------------
prediction = loaded_model.predict(img_array, verbose=0)
confidence  = float(prediction[0][0])               # sigmoid output in [0, 1]
label       = CLASS_NAMES[int(confidence >= 0.5)]   # threshold at 0.5

print(f"\n📸 Demo image : {os.path.basename(demo_image_path)}")
print(f"   Raw output  : {confidence:.4f}  (>= 0.5 → Parasitized)")
print(f"   Prediction  : {label}")
print(f"   Confidence  : {max(confidence, 1 - confidence) * 100:.1f}%")
