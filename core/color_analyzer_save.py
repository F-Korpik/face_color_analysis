import cv2
import numpy as np


def bgr_to_lab(bgr_color):
    # 1. Tworzymy jedno-pikselowy obraz dla OpenCV
    pixel = np.uint8([[bgr_color]])
    lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)

    # 2. Wyciągamy wartości
    l, a, b = lab_pixel[0][0]

    # 3. Przeskalowanie do standardowych jednostek
    # OpenCV Lab: L (0-255), a (0-255), b (0-255)
    # Standard Lab: L (0-100), a (-128-127), b (-128-127)
    l_std = l * 100 / 255
    a_std = a - 128
    b_std = b - 128

    return l_std, a_std, b_std


def calculate_eye_skin_contrast(eye_l, skin_l):
    """
    Oblicza różnicę jasności między okiem a skórą.
    """
    contrast = abs(skin_l - eye_l)

    if contrast > 40:
        return "HIGH CONTRAST"
    elif contrast < 20:
        return "LOW CONTRAST"
    else:
        return "MEDIUM CONTRAST"


class SkinAnalyzer:
    def __init__(self, skin_bgr):
        self.l, self.a, self.b = bgr_to_lab(skin_bgr)

    def get_temperature_stats(self):
        """
        Zwraca wynik temperatury:
        Wartości dodatnie -> Ciepły (dominuje żółty)
        Wartości ujemne -> Chłodny (dominuje różowy/czerwony)
        """
        # Prosta metoda: różnica między składową żółtą a czerwoną
        # W skórze typowo ciepłej b jest wyraźnie wyższe niż a
        temp_score = self.b - self.a

        if temp_score > 5:
            undertone = "WARM"
        elif temp_score < -2:
            undertone = "COOL"
        else:
            undertone = "NEUTRAL"

        return {
            "undertone": undertone,
            "score": round(temp_score, 2),
            "brightness": round(self.l, 2)
        }


class EyeAnalyzer:
    def __init__(self, iris_bgr):
        self.bgr = iris_bgr
        # Konwersja na LAB dla jasności
        self.l, self.a, self.b = bgr_to_lab(iris_bgr)
        # Konwersja na HSV dla saturacji (czystości)
        self.hsv = cv2.cvtColor(np.uint8([[iris_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    def get_eye_stats(self):
        saturation = self.hsv[1]  # Zakres 0-255 w OpenCV
        brightness = self.l  # Zakres 0-100

        # Logika chromatyczności
        if saturation > 80:  # Próg dla "czystych" oczu
            chroma = "BRIGHT"
        elif saturation < 40:
            chroma = "MUTED"
        else:
            chroma = "MEDIUM"

        # Logika głębi
        if brightness < 35:
            depth = "DEEP"
        elif brightness > 65:
            depth = "LIGHT"
        else:
            depth = "MEDIUM"

        return {
            "chroma": chroma,
            "depth": depth,
            "saturation_val": int(saturation),
            "brightness_val": round(brightness, 2)
        }


class ContrastAnalyzer:
    def __init__(self, skin_lab, eyebrows_lab):
        self.skin_l, self.skin_a, self.skin_b = skin_lab
        self.brows_l, self.brows_a, self.brows_b = eyebrows_lab

    def get_contrast_stats(self):
        # 1. Obliczamy różnicę jasności (Delta L)
        contrast_score = abs(self.skin_l - self.brows_l)

        # Klasyfikacja kontrastu
        if contrast_score > 35:
            contrast_level = "HIGH"
        elif contrast_score < 18:
            contrast_level = "LOW"
        else:
            contrast_level = "MEDIUM"

        # 2. Temperatura wtórna brwi (czy są złote/ciepłe czy szare/chłodne)
        # Patrzymy głównie na kanał b (żółty-niebieski)
        if self.brows_b > 8:
            brows_temp = "WARM"
        elif self.brows_b < 3:
            brows_temp = "COOL"
        else:
            brows_temp = "NEUTRAL"

        return {
            "contrast_score": round(contrast_score, 2),
            "contrast_level": contrast_level,
            "brows_temperature": brows_temp
        }


class LipAnalyzer:
    def __init__(self, lips_bgr):
        self.l, self.a, self.b = bgr_to_lab(lips_bgr)

    def get_lip_stats(self):
        # Stosunek żółci do czerwieni
        # Usta ciepłe mają b zbliżone do a lub przynajmniej b > 10
        # Usta chłodne mają a znacznie większe od b (często b < 5)

        if self.b > 12:
            undertone = "WARM"
        elif self.b < 7:
            undertone = "COOL"
        else:
            undertone = "NEUTRAL"

        return {
            "lip_undertone": undertone,
            "val_a": round(self.a, 2),
            "val_b": round(self.b, 2)
        }


class SeasonalAnalyzer:
    def __init__(self, medians):
        # Inicjalizacja analizatorów składowych
        self.skin = SkinAnalyzer(medians['skin'])
        self.eye = EyeAnalyzer(medians['iris'])
        self.lips = LipAnalyzer(medians['lips'])

        # Przygotowanie danych LAB do analizy kontrastu
        skin_lab = (self.skin.l, self.skin.a, self.skin.b)
        brows_lab = bgr_to_lab(medians['eyebrows'])
        self.contrast = ContrastAnalyzer(skin_lab, brows_lab)

    def predict_season(self):
        # 1. Pobieramy statystyki z każdego modułu
        skin_stats = self.skin.get_temperature_stats()
        eye_stats = self.eye.get_eye_stats()
        contrast_stats = self.contrast.get_contrast_stats()
        lip_stats = self.lips.get_lip_stats()

        # 2. Wyznaczamy dominującą temperaturę (Głosowanie: skóra + usta)
        temp_score = 0
        if skin_stats['undertone'] == "WARM": temp_score += 1
        if skin_stats['undertone'] == "COOL": temp_score -= 1
        if lip_stats['lip_undertone'] == "WARM": temp_score += 1
        if lip_stats['lip_undertone'] == "COOL": temp_score -= 1

        is_warm = temp_score > 0

        # 3. Logika przypisania do 4 głównych grup
        # Uproszczone drzewo decyzyjne
        if not is_warm:  # CHŁODNE (Lato / Zima)
            if contrast_stats['contrast_level'] == "HIGH" or eye_stats['chroma'] == "BRIGHT":
                return self._classify_winter(eye_stats, contrast_stats)
            else:
                return self._classify_summer(eye_stats, contrast_stats)
        else:  # CIEPŁE (Wiosna / Jesień)
            if contrast_stats['contrast_level'] == "HIGH" or eye_stats['chroma'] == "BRIGHT":
                return self._classify_spring(eye_stats, contrast_stats)
            else:
                return self._classify_autumn(eye_stats, contrast_stats)

    # Funkcje pomocnicze do doprecyzowania podtypu (12 sub-seasons)
    def _classify_winter(self, eye, contrast):
        if eye['chroma'] == "BRIGHT": return "Bright Winter"
        if eye['depth'] == "DEEP": return "Deep Winter"
        return "True Winter"

    def _classify_summer(self, eye, contrast):
        if eye['chroma'] == "MUTED": return "Soft Summer"
        if eye['depth'] == "LIGHT": return "Light Summer"
        return "True Summer"

    def _classify_spring(self, eye, contrast):
        if eye['chroma'] == "BRIGHT": return "Bright Spring"
        if eye['depth'] == "LIGHT": return "Light Spring"
        return "True Spring"

    def _classify_autumn(self, eye, contrast):
        if eye['chroma'] == "MUTED": return "Soft Autumn"
        if eye['depth'] == "DEEP": return "Deep Autumn"
        return "True Autumn"