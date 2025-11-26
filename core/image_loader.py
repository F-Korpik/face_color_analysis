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