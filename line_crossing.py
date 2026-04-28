"""
line_crossing.py
────────────────
Virtual line crossing detection for SafeRoad AI.

Tracks vehicles across frames using IoU-based matching and fires a
CrossingEvent the moment any vehicle's centre crosses the user-defined line.

Usage
-----
    tracker = LineCrossingTracker(line_pos=400, direction="down", axis="horizontal")

    for each frame:
        events = tracker.update(frame_rgb, vehicle_bboxes)
        for ev in events:
            # ev.bbox        — vehicle bbox at crossing moment  (x1,y1,x2,y2)
            # ev.track_id    — unique integer vehicle id
            # ev.frame_crop  — cropped vehicle region, RGB numpy array
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data-types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    bbox:     tuple   # (x1, y1, x2, y2)
    center:   tuple   # (cx, cy)  floats
    hits:     int = 1
    missed:   int = 0
    crossed:  bool = False   # True once this vehicle has already triggered

    @property
    def cx(self) -> float:
        return self.center[0]

    @property
    def cy(self) -> float:
        return self.center[1]


@dataclass
class CrossingEvent:
    track_id:   int
    bbox:       tuple        # (x1, y1, x2, y2)
    frame_crop: np.ndarray   # cropped vehicle ROI (RGB)
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center(bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-object tracker with line-crossing detection
# ─────────────────────────────────────────────────────────────────────────────

class LineCrossingTracker:
    """
    Parameters
    ----------
    line_pos   : pixel position of the line
                   horizontal axis → y-coordinate
                   vertical axis   → x-coordinate
    direction  : "down" | "up" | "right" | "left"
    axis       : "horizontal" | "vertical"
    iou_thresh : minimum IoU to associate a detection with an existing track
    max_missed : frames a track may be absent before being deleted
    """

    def __init__(
        self,
        line_pos:   int   = 400,
        direction:  str   = "down",
        axis:       str   = "horizontal",
        iou_thresh: float = 0.30,
        max_missed: int   = 10,
    ):
        self.line_pos   = line_pos
        self.direction  = direction
        self.axis       = axis
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed

        self._tracks:  dict[int, Track] = {}
        self._next_id: int = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_rgb:  np.ndarray,
        detections: list,            # list of (x1,y1,x2,y2)
    ) -> list:                       # list of CrossingEvent
        """
        Match detections → tracks, update positions, detect crossings.
        Returns a list of CrossingEvent for any vehicle that crossed this frame.
        """
        events = []
        unmatched = list(range(len(detections)))
        matched_ids: set = set()

        # ── Match existing tracks ─────────────────────────────────────────────
        for tid, track in list(self._tracks.items()):
            if not unmatched:
                break
            best_score = self.iou_thresh
            best_didx  = -1
            for didx in unmatched:
                score = _iou(track.bbox, detections[didx])
                if score > best_score:
                    best_score = score
                    best_didx  = didx
            if best_didx >= 0:
                prev_center  = track.center
                track.bbox   = detections[best_didx]
                track.center = _center(detections[best_didx])
                track.hits  += 1
                track.missed = 0
                matched_ids.add(tid)
                unmatched.remove(best_didx)

                # Line-crossing check
                if not track.crossed:
                    ev = self._check_crossing(track, prev_center, track.center, frame_rgb)
                    if ev:
                        track.crossed = True
                        events.append(ev)

        # ── Age unmatched tracks ──────────────────────────────────────────────
        dead = []
        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track.missed += 1
                if track.missed > self.max_missed:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        # ── Spawn new tracks for unmatched detections ─────────────────────────
        for didx in unmatched:
            bbox   = detections[didx]
            tid    = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=bbox,
                center=_center(bbox),
            )

        return events

    # ── Crossing test ──────────────────────────────────────────────────────────

    def _check_crossing(
        self,
        track:       Track,
        prev_center: tuple,
        curr_center: tuple,
        frame_rgb:   np.ndarray,
    ) -> Optional[CrossingEvent]:
        lp = self.line_pos

        if self.axis == "horizontal":
            prev_val = prev_center[1]   # y
            curr_val = curr_center[1]
        else:
            prev_val = prev_center[0]   # x
            curr_val = curr_center[0]

        crossed = False
        if self.direction in ("down", "right"):
            crossed = (prev_val < lp <= curr_val)
        elif self.direction in ("up", "left"):
            crossed = (prev_val > lp >= curr_val)

        if not crossed:
            return None

        # Crop vehicle ROI
        x1, y1, x2, y2 = track.bbox
        h, w = frame_rgb.shape[:2]
        crop = frame_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

        return CrossingEvent(
            track_id=track.track_id,
            bbox=track.bbox,
            frame_crop=crop,
        )

    @property
    def tracks(self) -> list:
        return list(self._tracks.values())

    def reset(self):
        self._tracks.clear()
        self._next_id = 0


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_virtual_line(
    image_rgb: np.ndarray,
    line_pos:  int,
    axis:      str  = "horizontal",
    triggered: bool = False,
    label:     str  = "Detection Line",
) -> np.ndarray:
    """
    Draw a dashed semi-transparent line on image_rgb.
    Yellow at rest → red when a crossing was detected this frame.
    """
    overlay = image_rgb.copy()
    h, w    = image_rgb.shape[:2]

    # Colours are in RGB (Streamlit displays RGB images)
    color = (255, 59, 59) if triggered else (245, 197, 24)
    thick = 3 if triggered else 2

    dash, gap = 20, 10

    if axis == "horizontal":
        y = min(max(line_pos, 0), h - 1)
        x = 0
        while x < w:
            cv2.line(overlay, (x, y), (min(x + dash, w), y), color, thick)
            x += dash + gap
        # Label tag
        tag = f"  {label}  "
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(overlay, (6, y - th - 10), (6 + tw + 4, y - 2), color, -1)
        cv2.putText(overlay, tag, (8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        x = min(max(line_pos, 0), w - 1)
        yp = 0
        while yp < h:
            cv2.line(overlay, (x, yp), (x, min(yp + dash, h)), color, thick)
            yp += dash + gap
        cv2.putText(overlay, label, (x + 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Blend for semi-transparency
    return cv2.addWeighted(overlay, 0.85, image_rgb, 0.15, 0)


def draw_vehicle_tracks(
    image_rgb:         np.ndarray,
    tracks:            list,
    events_this_frame: list,
    belt_results:      dict,   # track_id → seatbelt-result dict
) -> np.ndarray:
    """
    Colour-coded bounding boxes:
      YELLOW — detected, not yet crossed
      GREEN  — crossed + compliant
      RED    — crossed + violation  (shows plate number)
    """
    img = image_rgb.copy()
    crossing_ids = {ev.track_id for ev in events_this_frame}

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        tid = track.track_id
        res = belt_results.get(tid)

        if res is None:
            color = (245, 197, 24)    # yellow
            label = f"ID:{tid}"
        elif res.get("has_belt"):
            color = (0, 229, 160)     # green
            label = f"ID:{tid} ✓ SAFE"
        else:
            color = (255, 59, 59)     # red
            plate = res.get("plate") or "???"
            label = f"ID:{tid} ✗ {plate}"

        lw = 3 if (tid in crossing_ids or res) else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        if res:
            conf_lbl = f"conf {res.get('confidence', 0):.2f}"
            cv2.putText(img, conf_lbl, (x1 + 4, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Mock vehicle detector  (replace with real YOLO vehicle model)
# ─────────────────────────────────────────────────────────────────────────────

def mock_vehicle_detect(frame_rgb: np.ndarray) -> list:
    """
    Simulate bounding-box detections for vehicles.

    Replace with:
        results = vehicle_yolo.predict(frame_rgb, classes=[2,3,5,7], conf=0.5)
        return [tuple(map(int, b)) for b in results[0].boxes.xyxy.tolist()]
    """
    import random
    h, w = frame_rgb.shape[:2]
    seed = int(frame_rgb[::40, ::40].mean() * 100) % 9999
    rng  = random.Random(seed)
    n    = rng.randint(0, 3)
    bboxes = []
    for _ in range(n):
        x1 = rng.randint(20, max(21, w - 200))
        y1 = rng.randint(10, max(11, h - 180))
        bw = rng.randint(120, 240)
        bh = rng.randint(100, 200)
        bboxes.append((x1, y1, min(x1 + bw, w - 1), min(y1 + bh, h - 1)))
    return bboxes
