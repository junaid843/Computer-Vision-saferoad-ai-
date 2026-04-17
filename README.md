# 🚦 SafeRoad AI — Seatbelt Violation Detection System

Detects seatbelt violations in images/video using YOLOv8, extracts license plate numbers with OCR, logs violations to PostgreSQL, and displays everything in a Streamlit dashboard.

---

## Architecture

```
Input Image/Video
       ↓
 YOLOv8 Model 1  ──────────────────────────────────────────────
 (Seatbelt Detection)                                          │
   ├── Person_with_seatbelt    → mark as compliant             │
   └── Person_without_seatbelt → flag as violator              │
                                      ↓                        │
                         YOLOv8 Model 2                        │
                         (License Plate Detection)             │
                              ↓                                │
                    Find nearest plate to violator             │
                              ↓                                │
                    Crop plate region                          │
                              ↓                                │
                    Tesseract OCR → extract text               │
                              ↓                                │
                    PostgreSQL (violations table)              │
                              ↓                                │
                    Streamlit UI dashboard ←─────────────────── ┘
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <repo>
cd seatbelt_violation_system

pip install -r requirements.txt

# Install Tesseract binary
sudo apt install tesseract-ocr          # Ubuntu/Debian
# brew install tesseract                # macOS
```

### 2. Download datasets and train models

```bash
# Get your Roboflow API key from https://app.roboflow.com

# Train seatbelt detection model
python train/train_models.py \
    --task seatbelt \
    --api_key YOUR_ROBOFLOW_KEY \
    --epochs 50 \
    --imgsz 640

# Train license plate detection model
python train/train_models.py \
    --task plate \
    --api_key YOUR_ROBOFLOW_KEY \
    --epochs 50 \
    --imgsz 640
```

Best weights are saved to:
- `runs/train/seatbelt_yolov8/weights/best.pt`
- `runs/train/plate_yolov8/weights/best.pt`

Copy them to the `models/` directory:
```bash
cp runs/train/seatbelt_yolov8/weights/best.pt models/seatbelt_yolov8.pt
cp runs/train/plate_yolov8/weights/best.pt    models/license_plate_yolov8.pt
```

### 3. Set up PostgreSQL (optional)

```sql
CREATE DATABASE saferoad;
```

The app creates the `violations` table automatically on first connection.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Connecting Real Models

In `app.py`, replace the `mock_detect()` call with the real detector:

```python
# At the top of app.py, add:
from models.detector import SeatbeltDetector, annotate

detector = SeatbeltDetector(
    seatbelt_model_path="models/seatbelt_yolov8.pt",
    plate_model_path="models/license_plate_yolov8.pt",
    conf_threshold=0.60,
)

# Replace mock_detect(img_np) with:
detections = detector.detect(img_np)
annotated  = annotate(img_np, detections)
```

---

## Database Schema

```sql
CREATE TABLE violations (
    id             SERIAL PRIMARY KEY,
    plate_number   TEXT,
    violation_type TEXT        DEFAULT 'No Seatbelt',
    confidence     REAL,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    image_name     TEXT,
    frame_number   INT
);
```

---

## Dataset Links

| Dataset | URL |
|---------|-----|
| Seatbelt Detection | https://universe.roboflow.com/renuka-ed2ia/seat-belt-detection-c2nia |
| License Plate | https://universe.roboflow.com/manasa-n-n/car-license-plate-detection-maehj |

---

## Project Structure

```
seatbelt_violation_system/
├── app.py                   ← Streamlit UI (main entry point)
├── requirements.txt
├── README.md
├── models/
│   ├── detector.py          ← SeatbeltDetector class (YOLO + OCR)
│   ├── seatbelt_yolov8.pt   ← trained weights (add after training)
│   └── license_plate_yolov8.pt
├── db/
│   └── database.py          ← ViolationDB class (PostgreSQL)
└── train/
    └── train_models.py      ← training + validation scripts
```

---

## Deployment on HuggingFace Spaces

1. Create a new Space → SDK: Streamlit
2. Upload all files
3. Add `requirements.txt` (HuggingFace installs automatically)
4. Set `DB_URL` as a Space secret for PostgreSQL connection
5. Update `app.py` to read: `db_url = os.environ.get("DB_URL", "")`
