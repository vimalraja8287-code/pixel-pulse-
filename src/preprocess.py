"""Image preprocessing with OpenCV for blood smear images."""

import cv2
import numpy as np


def load_and_preprocess(image_path: str, target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Load image with OpenCV and preprocess for CNN input.
    - Read as RGB
    - Resize to target size
    - Normalize to [0, 1]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Simple augmentation: random flip (for use in training pipeline).
    Returns image in same format (float32 [0,1]).
    """
    if np.random.random() > 0.5:
        img = np.fliplr(img).copy()
    if np.random.random() > 0.5:
        img = np.flipud(img).copy()
    return img


def clahe_enhance(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast.
    Useful for microscopic images. Input: RGB float [0,1]; output: same.
    """
    img_uint8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.0
