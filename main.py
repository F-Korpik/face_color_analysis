import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.little_functions import load_image
from core.little_functions import resize_img
from core.face_detector import FaceAnalyzer
from core.median_colors import get_median_colors


TEST_IMAGE_PATH = os.path.join("data", "input", "test1.jpg")
img_height = 800


def run_face_analysis():
    print("--- Face analysis begins ---")
    print(f"Loading image from: {TEST_IMAGE_PATH}")

    loaded_image = load_image(TEST_IMAGE_PATH)

    if loaded_image is not None:
        image, img_width = resize_img(loaded_image, img_height)

    else:
        print("Analiza zakończona niepowodzeniem: Błąd ładowania obrazu.")
        return

    #Showing image
    try:
        # 2. Inicjalizacja i detekcja twarzy (Mediapipe)
        face_analyzer = FaceAnalyzer()
        landmarks = face_analyzer.detect_and_get_landmarks(image)

        if landmarks is None:
            print("Analiza zakończona: Nie wykryto twarzy.")
            return

        print(f"INFO: Wykryto {landmarks.shape[0]} punktów landmarks (Mediapipe).")


        # --- WIZUALIZACJA WYNIKÓW ---
        display_image = image.copy()
        h, w, _ = display_image.shape

        # Rysowanie punktów
        for (x, y) in landmarks:
            cv2.circle(display_image, (x, y), 1, (0, 0, 255), -1)


        medians, masks= get_median_colors(image, landmarks)

        canv_width = w + 110 + 5
        canv_height = h + 10
        display_canvas = np.full((canv_height, canv_width, 3), (255, 255, 255), dtype=np.uint8)

        display_canvas[5:h + 5, 110:110 + w] = display_image

        p_y = 20

        for key, color_value in medians.items():
            if color_value is not None:
                preview_square = np.full((100, 100, 3), color_value, dtype=np.uint8)
                display_canvas[p_y: (p_y + 100), 5: 105] = preview_square
                cv2.putText(display_canvas, key, (5, p_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                p_y += 120  # Przesunięcie w dół dla następnego koloru
                print(f"{key} - Mediana koloru (BGR): {color_value}")


        # Podgląd
        cv2.imshow('Analiza Kolorystyczna', display_canvas)

        # Wyświetlanie masek w osobnych oknach
        for key, mask_value in masks.items():
            if mask_value is not None:
                # f-string jest kluczowy dla unikalnych nazw okien
                cv2.imshow(f"Maska: {key}", mask_value)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("INFO: Obraz z landmarkami został wyświetlony i zamknięty.")

    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas detekcji: {e}")


    print("--- Analiza zakończona pomyślnie ---")
    pass



if __name__ == "__main__":
    run_face_analysis()