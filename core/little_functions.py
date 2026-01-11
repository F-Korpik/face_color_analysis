import cv2
import numpy as np
from typing import Optional

def load_image(file_path: str) -> Optional[np.ndarray]:
    if not file_path:
        return None

    img = cv2.imread(file_path)

    if img is None:
        print(f"Error: Unable to load image from {file_path}")
    else:
        print(f"INFO: Loaded image from {file_path} with size {img.shape}")

    return img


def resize_img(image: np.ndarray, new_height):
    h, w = image.shape[:2]
    scale = new_height / h
    new_width = int(w * scale)

    if new_width > 1200:
        scale = 1200 / new_width
        new_height = int(new_height * scale)
        new_width = int(new_width * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image, new_width


def get_center(landmarks, iris_indices):
    pts = landmarks[iris_indices].astype(np.int32)

    (x, y), radius = cv2.minEnclosingCircle(pts)
    center_x, center_y = (int(x), int(y))

    return center_x, center_y