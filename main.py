import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.image_loader import load_image


TEST_IMAGE_PATH = os.path.join("data", "input", "test_image.jpg")


def run_face_analysis():
    print("--- Face analysis begins ---")
    print(f"Loading image from: {TEST_IMAGE_PATH}")

    image = load_image(TEST_IMAGE_PATH)

    if image is None:
        print("Analiza zakończona niepowodzeniem: Błąd ładowania obrazu.")
        return

    #Showing image
    try:
        cv2.imshow('Face Analysis Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("INFO: Obraz został wyświetlony i zamknięty.")

    except cv2.error as e:
        print(f"ERROR: Błąd wyświetlania obrazu przez OpenCV. Upewnij się, że masz GUI (np. na serwerach Docker to może być problem): {e}")


    print("--- Face analysis ends ---")
    plt.imshow(image)

    pass



if __name__ == "__main__":
    run_face_analysis()