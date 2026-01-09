import cv2
import numpy as np
import math
from typing import Optional, Tuple

from core.little_functions import get_center


# Indeksy punktów charakterystycznych MediaPipe
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]


def mask_and_median(image, h, w, landmarks, base_areas, excluded_features):
    # 1. Tworzenie maski
    mask = np.zeros((h, w), dtype=np.uint8)
    for area in base_areas:
        area_points = landmarks[area].astype(np.int32)
        cv2.fillPoly(mask, [area_points], 255)

    # 2. Tworzenie maski obszarów wykluczonych
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    for feature_indices in excluded_features:
        feature_points = landmarks[feature_indices].astype(np.int32)
        cv2.fillPoly(exclude_mask, [feature_points], 255)

    # 3. Finalna maska
    final_mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(exclude_mask))

    # 4. Obliczanie MEDIANY
    mask_pixels = image[final_mask > 0]

    if mask_pixels.size == 0:
        return None, final_mask

    # Liczymy medianę wzdłuż osi 0 (dla każdego kanału B, G, R osobno)
    mask_median_bgr = np.median(mask_pixels, axis=0)

    # Konwersja na int (np.median zwraca floaty)
    median_color = (int(mask_median_bgr[0]), int(mask_median_bgr[1]), int(mask_median_bgr[2]))

    return median_color, final_mask


def get_iris_median(image, landmarks, iris_indices, angle):

    pts = landmarks[iris_indices].astype(np.int32)

    (x, y), radius = cv2.minEnclosingCircle(pts)
    center_x, center_y = (int(x), int(y))
    r = int(radius)

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    width = 2 * r
    height = r

    rotated_rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)
    cv2.drawContours(mask, [box], 0, 255, -1)

    pupil_radius = int(r * 0.4)
    cv2.circle(mask, (center_x, center_y), pupil_radius, 0, -1)

    iris_circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_circle_mask, (center_x, center_y), int(r * 0.85), 255, -1)

    final_iris_mask = cv2.bitwise_and(mask, iris_circle_mask)

    iris_pixels = image[final_iris_mask > 0]
    if iris_pixels.size == 0:
        return None, final_iris_mask
    median_bgr = np.median(iris_pixels, axis=0)
    return (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])), final_iris_mask


def eyes_mask_and_median(image, landmarks, left_eye, right_eye):
    l_x, l_y = get_center(landmarks, left_eye)
    r_x, r_y = get_center(landmarks, right_eye)

    delta_y = l_y - r_y
    delta_x = l_x - r_x

    angle = math.degrees(math.atan2(delta_y, delta_x))


    left_color, left_m = get_iris_median(image, landmarks, left_eye, angle)
    right_color, right_m = get_iris_median(image, landmarks, right_eye, angle)

    if left_color and right_color:
        avg_eye = (np.array(left_color) + np.array(right_color)) // 2
        eyes_color = (int(avg_eye[0]), int(avg_eye[1]), int(avg_eye[2]))
    else:
        eyes_color = left_color or right_color

    eyes_mask = cv2.bitwise_or(left_m, right_m)

    return eyes_color, eyes_mask


def get_median_colors(image: np.ndarray, landmarks: np.ndarray):
    """
    Oblicza medianę koloru skóry (BGR), wykluczając oczy, brwi i usta.
    """
    if landmarks is None:
        return None, None

    h, w, _ = image.shape

    skin_mask_exclude = [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER, LIPS_INNER]
    lips_mask_exclude = [LIPS_INNER]

    skin_median, skin_mask = mask_and_median(image, h, w, landmarks, [FACE_OVAL], skin_mask_exclude)
    lips_median, lips_mask = mask_and_median(image, h, w, landmarks, [LIPS_OUTER], lips_mask_exclude)
    eybrows_median, eybrows_mask = mask_and_median(image,h, w, landmarks, [LEFT_EYEBROW, RIGHT_EYEBROW], [])
    eyes_median, eyes_mask = eyes_mask_and_median(image, landmarks, LEFT_IRIS, RIGHT_IRIS)


    medians = {"skin": skin_median, "lips": lips_median, "eybrows": eybrows_median, "iris":eyes_median}
    masks = {"skin": skin_mask, "lips": lips_mask, "eybrows": eybrows_mask, "iris": eyes_mask}

    return medians, masks