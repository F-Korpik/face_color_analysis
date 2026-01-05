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


def resize_img(image: np.ndarray, new_height) -> np.ndarray:
    """
    Przeskalowuje obraz, aby jego wysokość nie przekraczała max_height,
    zachowując przy tym proporcje szerokości.
    """
    h, w = image.shape[:2]
    scale = new_height / h
    new_width = int(w * scale)
        # INTER_AREA jest najlepszą metodą do zmniejszania obrazów
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image