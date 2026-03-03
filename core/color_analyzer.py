import cv2
import numpy as np
import math


def bgr_to_lab(bgr_color):
    pixel = np.array([[bgr_color]], dtype=np.float32) / 255.0
    lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)
    l, a, b = lab_pixel[0][0]

    # l_std = l * 100 / 255
    # a_std = a - 128
    # b_std = b - 128
    #
    # return l_std, a_std, b_std

    return l, a, b


class ColorAnalyzer:
    #1. Temperature
    WARM_THRESHOLD = 53.0
    COOL_THRESHOLD = 47.0

    #2a. Contrast
    CONTRAST_HIGH = 35.0  # Wyraźna oprawa (Zimy, Ciemne Jesienie)
    CONTRAST_LOW = 20.0  # Delikatna oprawa (Lata, Jasne Wiosny)

    #2b. Nasycenie tęczówki
    CHROMA_BRIGHT = 19  # "Czyste" oko (Wiosny, Zimy)
    CHROMA_MUTED = 8.0  # "Zgaszone" oko (Lata, Jesienie)



    def __init__(self, medians):
        self.medians = medians
        self.lab_data = {key: bgr_to_lab(val) for key, val in medians.items()}


    def skin_temperature_analyzer(self):
        #lightness, green-red, blue-yellow
        l, a, b = self.lab_data["skin"]

        hue_angle = math.degrees(math.atan2(b, a))

        if hue_angle > self.WARM_THRESHOLD: temp = "warm"
        elif hue_angle < self.COOL_THRESHOLD: temp = "cool"
        else: temp = "neutral"

        hue_angle = round(hue_angle, 2)

        return {
                "hue_angle": hue_angle,
                "temperature": temp
        }


    def lips_analyzer(self):
        l_skin, a_skin, b_skin = self.lab_data["skin"]
        l_lips, a_lips, b_lips = self.lab_data["lips"]

        # Temperatura ust (Kąt Hue dla ust)
        # Usta chłodne (malinowe) mają niższy kąt niż usta ciepłe (łososiowe)
        lips_hue = math.degrees(math.atan2(b_lips, a_lips))

        # Kontrast ust (czy są wyraźne)
        lips_contrast = l_skin - l_lips

        # Progi dla ust (orientacyjne - wymagają testów)
        # Zazwyczaj usta < 35 stopni są chłodne, > 40 są ciepłe
        if lips_hue > 38:
            lips_temp = "warm"
        elif lips_hue < 32:
            lips_temp = "cool"
        else:
            lips_temp = "neutral"

        return {
            "lips_hue": round(lips_hue, 2),
            "lips_temp": lips_temp,
            "lips_contrast": round(lips_contrast, 2)
        }


    def eye_and_contrast_analyzer(self):
        l_skin, a_skin, b_skin = self.lab_data["skin"]
        l_brows, a_brows, b_brows = self.lab_data["eyebrows"]
        l_iris, a_iris, b_iris = self.lab_data["iris"]

        contrast_val = l_skin - l_brows

        if contrast_val > self.CONTRAST_HIGH: contrast = "high"
        elif contrast_val < self.CONTRAST_LOW: contrast = "low"
        else: contrast = "medium"


        chroma_val = math.sqrt(float(a_iris)**2 + float(b_iris)**2)

        if chroma_val > self.CHROMA_BRIGHT: chroma = "bright"
        elif chroma_val < self.CHROMA_MUTED: chroma = "muted"
        else: chroma = "neutral"

        return {
            "contrast_score": round(contrast_val, 2),
            "contrast_type": contrast,
            "chroma_score": round(chroma_val, 2),
            "chroma_type": chroma
        }


    def get_preliminary_season(self):
        # poprawić zgodnie z chatem

        temp_results = self.skin_temperature_analyzer()
        eye_results = self.eye_and_contrast_analyzer()
        lips_results = self.lips_analyzer()

        skin_hue = temp_results["hue_angle"]
        lips_hue = lips_results["lips_hue"]
        l_skin = self.lab_data["skin"][0]

        contrast_type = eye_results["contrast_type"]
        contrast_val = float(eye_results["contrast_score"])
        chroma_type = eye_results["chroma_type"]
        chroma_val = float(eye_results["chroma_score"])

        # --- OBLICZANIE BIASU (Współczynnik temperatury) ---
        bias = (skin_hue - 50.0) + 1.5 * (lips_hue - 35.0)

        # --- LOGIKA 12 TYPÓW (Hierarchia atrybutów) ---

        # 1. PRIORYTET JASNOŚCI (Dla typów o niskim kontraście - np. Twoja modelka)
        # Jeśli kontrast jest niski i skóra bardzo jasna, to typ "Light" ma pierwszeństwo przed "Bright"
        if l_skin > 75 and contrast_val < 22:
            season = "Light Spring" if bias > 0 else "Light Summer"

        # 2. WYJĄTEK DLA WYSOKIEGO KONTRASTU (Typy Czyste/Zimowe - np. Daddario)
        elif contrast_type == "high" and lips_results["lips_temp"] == "cool" and bias < 5:
            # Obniżony próg dla "Bright", by wyłapać kontrastowe zimy o jasnych oczach
            season = "Bright Winter" if chroma_val > 14 else "True Winter"

        # 3. Dominanta: CHROMATYKA (Jeśli nie wpadliśmy w typy Light/High Contrast)
        elif chroma_type == "bright":
            season = "Bright Spring" if bias > 0 else "Bright Winter"
        elif chroma_type == "muted":
            season = "Soft Autumn" if bias > 0 else "Soft Summer"

        # 4. Dominanta: GŁĘBIA (Ciemna skóra/Wysoki kontrast przy średniej jasności)
        elif l_skin < 50 or (contrast_type == "high" and l_skin < 60):
            season = "Deep Autumn" if bias > 0 else "Deep Winter"

        # 5. Dominanta: TEMPERATURA (Typy "True" i pozostałe)
        else:
            if bias > 7.0:
                season = "True Spring" if l_skin > 65 else "True Autumn"
            elif bias < -7.0:
                season = "True Winter" if contrast_type == "high" else "True Summer"
            else:
                # Typy neutralne "Flow"
                if bias > 0:
                    season = "Warm Spring" if l_skin > 65 else "Warm Autumn"
                else:
                    season = "Cool Summer" if l_skin > 65 else "Cool Winter"

        return {
            "season": season,
            "details": {
                **temp_results,
                "eyes": eye_results,
                "lips": lips_results,
                "bias": round(bias, 2),
                "l_skin": round(l_skin, 2)
            }
        }
