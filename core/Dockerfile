# Użycie oficjalnego obrazu Pythona jako bazy
# Wybieramy wersję, która jest często używana z OpenCV
FROM python:3.9-slim

# Ustawienie zmiennej środowiskowej, aby Python nie buforował wyjścia
# Ułatwia to debugowanie w kontenerze
ENV PYTHONUNBUFFERED 1

# Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

# Kopiowanie pliku zależności i instalacja
# Używamy tej kolejności, aby wykorzystać cache Dockera – jeśli zależności się nie zmienią,
# ta warstwa nie będzie odbudowywana
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie reszty plików projektu do kontenera
# W tym: core/, main.py, data/models/ (wagi modeli)
COPY . /app

# OSTRZEŻENIE: OpenCV i Mediapipe mogą wymagać pakietów systemowych, np. dla wyświetlania.
# Jeśli w przyszłości wystąpią błędy z bibliotekami GUI/X-server,
# należy zmienić obraz na pełniejszy (np. python:3.9-buster) i dodać pakiety systemowe.

# Definicja domyślnego polecenia uruchamiającego aplikację
CMD ["python", "main.py"]