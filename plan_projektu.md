# 1. Wykrywanie Twarzy i Kluczowych Punktów (Landmarks)
## Implementacja:
	Zamiast samego wykrywania twarzy (Face Detection, np. kaskady Haara lub proste modele DL/SSD), sugeruję użycie Face Landmark Detection (np. biblioteka dlib lub zaawansowane funkcje OpenCV).
	
## Wartość Dodana:
	Landmarki (ok. 68 lub 81 punktów) precyzyjnie zdefiniują granice oczu, ust, linii włosów i konturu twarzy. Umożliwi to dokładne wyizolowanie ROI, a nie tylko prostokąta Bounding Box. Jest to kluczowe dla dokładności analizy kolorystycznej.
	
# 2. Metryka Koloru: Średnia czy Mediana?
## Mediana Koloru:
	Jest dobrym wyborem, ponieważ jest odporna na outliery (np. drobne refleksy światła, zanieczyszczenia na zdjęciu, pieprzyki, małe niedoskonałości).
## Lepsza Sugestia (Robustność):
	Zamiast tylko mediany, rozważ obliczenie dominującego koloru (np. za pomocą algorytmu K-Means Clustering). W danym ROI (np. skóra), K-Means z $K=1$ lub $K=3$ może wyłuskać najbardziej reprezentatywny kolor, ignorując szumy i dając Ci dodatkowo informację o wariancji (odcieniach).
	
# 3. Przestrzeń Barw
## Pamiętaj:
	Domyślnie OpenCV używa BGR, ale dla analizy kolorystycznej lepiej jest przekształcić ROI na przestrzeń HSV lub LAB.#### 
#### * HSV (Hue, Saturation, Value):
	Idealna, ponieważ Hue (barwa) jest oddzielona od intensywności (Value) i czystości (Saturation). Ułatwia to np. obiektywne określenie "tonacji" (ciepła/chłodna) skóry.
#### * LAB:
	Bardziej spójna z ludzkim postrzeganiem kolorów, dobra do precyzyjnej analizy różnic.
	
# Plan Projektu: Modułowa Architektura(Minimum Viable Product):
## Faza 1: Ustawienie Środowiska i Wstępne Przetwarzanie (MVP)
### 1.Repozytorium Git:
	Utwórz repozytorium (podstawowa higiena kodu).
### 2.Środowisko Python/Zależności:

	Użyj Virtual Environment i pliku requirements.txt (opencv-python, numpy, dlib - jeśli się zdecydujesz na landmarki).
	
### 3.Struktura Plików:
	/face_analysis/
    - main.py             # Logika uruchomienia
    - core/
        - image_loader.py # Ładowanie obrazu
        - face_detector.py# Moduł odpowiedzialny za wykrywanie i landmarki
    - data/
        - input/          # Katalog na zdjęcia wejściowe
        - output/         # Katalog na wyniki/wizualizacje


## Faza 2: Detekcja i Izolacja ROI:
### 1.Wykrywanie:
	Implementacja funkcji w core/face_detector.py do ładowania modelu detekcji twarzy i zwracania współrzędnych twarzy i/lub landmarków.
	
### 2.Normalizacja Obrazu:
	Opcjonalny krok: Wyrównanie i standaryzacja twarzy (Face Alignment) w oparciu o pozycję oczu, by zminimalizować wpływ kąta zdjęcia na analizę.
	
### 3.Definicja ROI:
	Stworzenie funkcji do wyciągania konkretnych masek dla ROI:
	* Skóra Twarzy: Kontur twarzy minus usta, oczy, brwi (kluczowe ROI).
	* Usta, Oczy, Włosy: Wyizolowanie masek na podstawie grup landmarków.


## Faza 3: Analiza Kolorystyczna i Metryki Danych

### 1. Moduł Analityczny:
	Stworzenie pliku core/color_analyzer.py.
	
### 2.Obliczenia:
	W tym module implementacja funkcji:Konwersja:
	* Konwersja ROI z BGR na HSV/LAB.
	* Metryka: Obliczenie mediany koloru i/lub dominującego koloru (K-Means) dla        każdego ROI.
	* Wyniki: Zwrócenie wyniku jako struktury danych (np. słownik/JSON/obiekt    Pandas) zawierającej ROI i odpowiadające im metryki koloru (np. {'skin': [H, S, V], 'lips': [H, S, V], ...}).


## Faza 4: Prezentacja i Wizualizacja

### 1.Wizualizacja:
	* Funkcja w main.py lub osobnym module do:Narysowania Bounding Boxów lub Landmarków na oryginalnym zdjęciu.Wizualizacji wyników: np. małe, kolorowe kwadraty obok zdjęcia, reprezentujące obliczony dominujący kolor dla każdego ROI.
	* Output Danych: Zapisanie danych analitycznych (JSON/CSV) do folderu data/output/.