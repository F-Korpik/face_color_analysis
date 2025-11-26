import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional

mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]

class FaceAnalyzer:
    """
    Detecting faces and landmarks using Mediapipe Face Mesh
    """

    def __init__(self):

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
        print("INFO: Mediapipe Face Mesh initialized.")


    def detect_and_get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("WARNING: Nie wykryto twarzy na zdjęciu.")
            return None

        # 3. Wybieramy pierwszą twarz i konwertujemy landmarki
        landmarks_data = results.multi_face_landmarks[0]

        h, w, c = image.shape
        coords = np.zeros((len(landmarks_data.landmark), 2), dtype=int)

        # Iteracja po punktach i skalowanie ich do pikseli (Mediapipe zwraca normalizowane [0, 1])
        for i, landmark in enumerate(landmarks_data.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            coords[i] = (x, y)

        return coords
