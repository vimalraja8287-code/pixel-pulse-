"""CNN model for malaria parasite detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(
    input_shape=(128, 128, 3),
    num_classes=2,
    dropout_rate=0.4,
    use_augmentation=True,
):
    """
    Build a CNN for binary classification: Parasitized vs Uninfected.
    use_augmentation: add RandomFlip/RandomRotation (only applied at training time).
    """
    blocks = [
        layers.Input(shape=input_shape),
    ]
    if use_augmentation:
        blocks.extend([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
        ])
    blocks.extend([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model = keras.Sequential(blocks, name="paradetect_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
