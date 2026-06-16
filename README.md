# Facial Color Analysis Tool

Python tool for automated personal color season analysis from a photo. Detects facial features using MediaPipe, extracts median colors from skin, eyes, lips, and eyebrows, then classifies the result into one of 12 seasonal color types (e.g. Soft Autumn, True Winter, Bright Spring).

![Demo](docs/screenshot.png)
<!-- Replace with an actual screenshot of the analysis output -->

---

## What It Does

1. Loads a photo and detects facial landmarks using **MediaPipe Face Mesh**
2. Extracts median BGR colors from four facial regions: skin, iris, lips, eyebrows
3. Converts colors to **CIE Lab color space** for perceptually accurate analysis
4. Analyzes skin temperature (warm/cool), eye chroma, skin-eyebrow contrast, and lip undertone
5. Classifies the result into one of **12 seasonal subtypes** based on a decision tree

---

## Tech Stack

**Language:** Python  
**Computer Vision:** OpenCV · MediaPipe  
**Color Analysis:** CIE Lab color space · HSV  
**Data:** NumPy · custom mask-based median extraction  
**Containerization:** Docker

---

## Project Structure

```
core/
├── face_detector.py       # MediaPipe Face Mesh wrapper
├── median_colors.py       # Facial region masking and median color extraction
├── color_analyzer.py      # CIE Lab analysis + 12-season classification
├── little_functions.py    # Image loading, resizing, geometry helpers
├── Dockerfile
└── docker-compose.yml
data/
├── input/                 # Input images
└── models/
main.py                    # Entry point
requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/F-Korpik/facial-color-analysis
cd facial-color-analysis

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Place your photo in `data/input/` and update the path in `main.py`:

```python
TEST_IMAGE_PATH = os.path.join("data", "input", "your_photo.jpg")
```

Then run:

```bash
python main.py
```

A window will open showing the analyzed face with color swatches for each region and the detected season printed to the console.

---

## How the Analysis Works

The classification uses **CIE Lab color space** rather than raw RGB, which better reflects how human perception responds to color differences.

Four metrics drive the decision tree:

| Metric | Method |
|--------|--------|
| Skin temperature | Hue angle in Lab (warm vs. cool undertone) |
| Eye chroma | Chroma value `√(a² + b²)` of iris |
| Facial contrast | Lightness delta between skin and eyebrows |
| Lip undertone | Hue angle of lips vs. skin |

A weighted bias score combines skin and lip temperature to determine warm/cool dominance, then contrast and chroma narrow the result to one of 12 subtypes.

---

## Seasonal Types Supported

| Warm | Cool |
|------|------|
| True Spring · Bright Spring · Light Spring · Warm Spring | True Summer · Soft Summer · Light Summer · Cool Summer |
| True Autumn · Soft Autumn · Deep Autumn · Warm Autumn | True Winter · Bright Winter · Deep Winter · Cool Winter |
