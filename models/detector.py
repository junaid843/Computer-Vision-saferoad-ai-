"""
models/detector.py
──────────────────
Real YOLO + OCR inference pipeline.
Replace mock_detect() in app.py with detect() from this module once models are trained.
"""

from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

# ── Optional imports (install before using) ──────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class SeatbeltDetector:
    """
    Two-stage detector:
      Stage 1 → YOLOv8 seatbelt model  (Person_with_seatbelt / Person_without_seatbelt)
      Stage 2 → YOLOv8 license plate model  (License_plate)
      Stage 3 → Tesseract OCR on cropped plate region
    """

    # Class indices from seatbelt model (adjust to match your label order)
    CLASS_WITH_BELT    = 0   # "Person_with_seatbelt"
    CLASS_WITHOUT_BELT = 1   # "Person_without_seatbelt"

    def __init__(
        self,
        seatbelt_model_path: str = "models/seatbelt_yolov8.pt",
        plate_model_path:    str = "models/license_plate_yolov8.pt",
        conf_threshold:      float = 0.60,
        iou_threshold:       float = 0.45,
    ):
        self.conf = conf_threshold
        self.iou  = iou_threshold

        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        # Load models
        self.seatbelt_model = YOLO(seatbelt_model_path)
        self.plate_model    = YOLO(plate_model_path)

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Run the full pipeline on a single RGB frame.

        Returns a list of detection dicts:
        {
            "bbox":       (x1, y1, x2, y2),
            "has_belt":   bool,
            "confidence": float,
            "plate":      str | None,
            "plate_bbox": (x1, y1, x2, y2) | None,
        }
        """
        results = []

        # ── Stage 1: Detect persons ───────────────────────────────────────────
        sb_results = self.seatbelt_model.predict(
            image_rgb,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        persons = []
        for box in sb_results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            has_belt = (cls_id == self.CLASS_WITH_BELT)
            persons.append({
                "bbox": (x1, y1, x2, y2),
                "has_belt": has_belt,
                "confidence": round(conf, 3),
                "plate": None,
                "plate_bbox": None,
            })

        if not persons:
            return []

        # ── Stage 2: Detect license plates (only if violations present) ───────
        violators = [p for p in persons if not p["has_belt"]]
        if violators:
            plate_results = self.plate_model.predict(
                image_rgb,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )[0]

            plates = []
            for box in plate_results.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                plates.append((px1, py1, px2, py2))

            # ── Stage 3: Assign nearest plate to each violator ────────────────
            for person in violators:
                if not plates:
                    break
                nearest_plate = self._nearest_plate(person["bbox"], plates)
                if nearest_plate:
                    person["plate_bbox"] = nearest_plate
                    crop = self._crop(image_rgb, nearest_plate)
                    person["plate"] = self._ocr(crop)

        results.extend(persons)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _nearest_plate(
        person_bbox: tuple[int, int, int, int],
        plates:      list[tuple[int, int, int, int]],
    ) -> Optional[tuple[int, int, int, int]]:
        """Return the plate bounding box nearest to the given person bbox."""
        px1, py1, px2, py2 = person_bbox
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        best_plate = None
        best_dist  = float("inf")
        for plate in plates:
            lx1, ly1, lx2, ly2 = plate
            lcx = (lx1 + lx2) / 2
            lcy = (ly1 + ly2) / 2
            dist = ((pcx - lcx) ** 2 + (pcy - lcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist  = dist
                best_plate = plate

        return best_plate

    @staticmethod
    def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _ocr(plate_crop: np.ndarray) -> Optional[str]:
        """Apply Tesseract OCR to extract plate number."""
        if not TESSERACT_AVAILABLE:
            return "OCR_UNAVAILABLE"
        if plate_crop.size == 0:
            return None

        # Pre-process for better OCR accuracy
        gray    = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        text = pytesseract.image_to_string(thresh, config=config).strip()
        text = "".join(c for c in text if c.isalnum() or c == "-")
        return text if len(text) >= 3 else None


# ─── Annotation Utility ───────────────────────────────────────────────────────
def annotate(image_rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the image."""
    img = image_rgb.copy()
    BELT_COLOR    = (0, 229, 160)   # green
    VIOL_COLOR    = (59, 59, 255)   # red (BGR)
    PLATE_COLOR   = (245, 197, 24)  # yellow

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = BELT_COLOR if det["has_belt"] else VIOL_COLOR
        label = "With Belt" if det["has_belt"] else "VIOLATION"
        conf  = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if det.get("plate_bbox"):
            px1, py1, px2, py2 = det["plate_bbox"]
            cv2.rectangle(img, (px1, py1), (px2, py2), PLATE_COLOR, 2)
            if det.get("plate"):
                cv2.putText(img, det["plate"], (px1, py1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, PLATE_COLOR, 1, cv2.LINE_AA)

    return img
