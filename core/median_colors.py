import cv2
import numpy as np
from typing import Optional, Tuple

# Indeksy punktów charakterystycznych MediaPipe
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def get_median_skin_color(image: np.ndarray, landmarks: np.ndarray) -> Tuple[Optional[Tuple[int, int, int]], Optional[np.ndarray]]:
    """
    Oblicza medianę koloru skóry (BGR), wykluczając oczy, brwi i usta.
    """
    if landmarks is None:
        return None, None

    h, w, _ = image.shape

    # 1. Tworzenie maski twarzy
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_points = landmarks[FACE_OVAL].astype(np.int32)
    cv2.fillPoly(face_mask, [face_points], 255)

    # 2. Tworzenie maski obszarów wykluczonych (oczy, brwi, usta)
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    features_to_exclude = [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER, LIPS_INNER]
    for feature_indices in features_to_exclude:
        feature_points = landmarks[feature_indices].astype(np.int32)
        cv2.fillPoly(exclude_mask, [feature_points], 255)

    # 3. Finalna maska skóry
    skin_mask = cv2.bitwise_and(face_mask, face_mask, mask=cv2.bitwise_not(exclude_mask))

    # 4. Obliczanie MEDIANY
    # Wybieramy tylko te piksele z obrazu, gdzie maska jest biała (> 0)
    # image[skin_mask > 0] zwraca tablicę o kształcie (liczba_pikseli, 3)
    skin_pixels = image[skin_mask > 0]

    if skin_pixels.size == 0:
        return None, skin_mask

    # Liczymy medianę wzdłuż osi 0 (dla każdego kanału B, G, R osobno)
    median_color_bgr = np.median(skin_pixels, axis=0)

    # Konwersja na int (np.median zwraca floaty)
    median_color = (int(median_color_bgr[0]), int(median_color_bgr[1]), int(median_color_bgr[2]))

    return median_color, skin_mask