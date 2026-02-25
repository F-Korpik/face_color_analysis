import cv2
import numpy as np
import math

from core.little_functions import get_center, create_rotated_rect

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


def get_iris_median(image, landmarks, iris_indices, angle, debug_image=None):
    #ściąć dolne rogi na trójkątno do środka
    pts = landmarks[iris_indices].astype(np.int32)

    (x, y), radius = cv2.minEnclosingCircle(pts)
    center_x, center_y = (int(x), int(y))
    r = int(radius)


    # --- KOREKTA NA NAJCIEMNIEJSZY PUNKT (ŹRENICĘ) ---
    roi_size = max(2, r // 2)
    y_start, y_end = max(0, center_y - roi_size), min(image.shape[0], center_y + roi_size)
    x_start, x_end = max(0, center_x - roi_size), min(image.shape[1], center_x + roi_size)

    roi = image[y_start:y_end, x_start:x_end]
    if roi.size > 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Znajdujemy najciemniejszy punkt (źrenicę)
        _, _, min_loc, _ = cv2.minMaxLoc(gray_roi)
        # Aktualizujemy środek
        center_x = x_start + min_loc[0]
        center_y = y_start + min_loc[1]

    # --- CIĄG DALSZY LOGIKI MASKI ---

    h, w = image.shape[:2]
    iris_mask = np.zeros((h, w), dtype=np.uint8)

    width = int(1.25 * r)
    useful_r = int(r * 0.8)
    height = r

    rotated_rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)
    cv2.drawContours(iris_mask, [box], 0, 255, -1)

    #sam obrócony prostokąt


    pupil_radius = int(r * 0.3)
    cv2.circle(iris_mask, (center_x, center_y), pupil_radius, 0, -1)

    #obrócony prostokąt z dziurą na źrenice

    # Obcinanie górnych 40%

    cutoff_dist = 0.2 * r # to jest "pionowa" odległość od środka źrenicy

    cutof_size = 10 * r

    pts1 = np.array([
        [-cutof_size, -cutof_size],  # Lewy górny
        [cutof_size, -cutof_size],  # Prawy górny
        [cutof_size, -cutoff_dist],  # Prawy dolny (linia cięcia)
        [-cutof_size, -cutoff_dist]  # Lewy dolny (linia cięcia)
    ])

    M = cv2.getRotationMatrix2D((0, 0), 180 - angle, 1)
    rotated_pts = cv2.transform(np.array([pts1]), M)[0]
    translated_pts = rotated_pts + [center_x, center_y]

    # 5. Wypełniamy ten obszar zerami na finalnej masce
    cv2.fillPoly(iris_mask, [translated_pts.astype(np.int32)], 0)

    #odcięcie góry zgodnie z cutoff_dist


    final_iris_mask = iris_mask


    iris_pixels = image[final_iris_mask > 0]
    if iris_pixels.size == 0:
        return None, final_iris_mask

    median_bgr = np.median(iris_pixels, axis=0)

    # #Rysowanie okręgu w celach diagnostycznych
    # if debug_image is not None:
    #     cv2.circle(debug_image, (center_x, center_y), 2, (0, 255, 0), -1)  # Środek źrenicy
    #     cv2.circle(debug_image, (center_x, center_y), useful_r, (255, 255, 0), 1)  # Zakres tęczówki

    overlay = image.copy()
    overlay[final_iris_mask > 0] = [0, 255, 0]

    alpha = 0.4
    debug_image_combined = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imshow("Diagnostyka Maski", debug_image_combined)

    return (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])), final_iris_mask


def eyes_mask_and_median(image, landmarks, left_eye, right_eye, debug_image=None):
    l_x, l_y = get_center(landmarks, left_eye)
    r_x, r_y = get_center(landmarks, right_eye)

    delta_y = l_y - r_y
    delta_x = l_x - r_x

    angle = math.degrees(math.atan2(delta_y, delta_x))

    left_color, left_m = get_iris_median(image, landmarks, left_eye, angle, debug_image)
    right_color, right_m = get_iris_median(image, landmarks, right_eye, angle, debug_image)

    if left_color and right_color:
        avg_eye = (np.array(left_color) + np.array(right_color)) // 2
        eyes_color = (int(avg_eye[0]), int(avg_eye[1]), int(avg_eye[2]))
    else:
        eyes_color = left_color or right_color

    eyes_mask = cv2.bitwise_or(left_m, right_m)

    return eyes_color, eyes_mask


def get_median_colors(image: np.ndarray, landmarks: np.ndarray, debug_image=None):
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
    eyebrows_median, eyebrows_mask = mask_and_median(image,h, w, landmarks, [LEFT_EYEBROW, RIGHT_EYEBROW], [])
    eyes_median, eyes_mask = eyes_mask_and_median(image, landmarks, LEFT_IRIS, RIGHT_IRIS, debug_image)


    medians = {"skin": skin_median, "lips": lips_median, "eyebrows": eyebrows_median, "iris":eyes_median}
    masks = {"skin": skin_mask, "lips": lips_mask, "eyebrows": eyebrows_mask, "iris": eyes_mask}

    return medians, masks