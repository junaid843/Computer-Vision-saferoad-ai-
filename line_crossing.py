"""
line_crossing.py
────────────────
Virtual line crossing detection for SafeRoad AI.

Tracks vehicles across frames and triggers seatbelt + plate inspection
the moment any vehicle crosses a user-defined line.

Usage
-----
    tracker = LineCrossingTracker(line_pos=400, direction="down", axis="horizontal")

    for each frame:
        events = tracker.update(frame_rgb, vehicle_bboxes)
        for ev in events:
            # ev["bbox"]        — vehicle bbox at crossing moment
            # ev["track_id"]    — unique vehicle id
            # ev["frame_crop"]  — cropped vehicle region (numpy RGB)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id:   int
    bbox:       tuple[int, int, int, int]   # (x1, y1, x2, y2)
    center:     tuple[float, float]
    hits:       int = 1
    missed:     int = 0
    crossed:    bool = False                # has this vehicle already triggered?

    @property
    def cx(self) -> float:
        return self.center[0]

    @property
    def cy(self) -> float:
        return self.center[1]


@dataclass
class CrossingEvent:
    track_id:   int
    bbox:       tuple[int, int, int, int]
    frame_crop: np.ndarray                  # cropped vehicle ROI
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _center(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Simple IoU-based multi-object tracker
# ─────────────────────────────────────────────────────────────────────────────

class LineCrossingTracker:
    """
    Parameters
    ----------
    line_pos   : pixel position of the line (y-coord if horizontal, x-coord if vertical)
    direction  : "down" | "up" | "right" | "left"
    axis       : "horizontal" | "vertical"
    iou_thresh : minimum IoU to associate a detection with an existing track
    max_missed : frames a track may go undetected before deletion
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

        self._tracks:   dict[int, Track] = {}
        self._next_id:  int = 0
        self._prev_centers: dict[int, tuple[float, float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame_rgb:   np.ndarray,
        detections:  list[tuple[int, int, int, int]],   # list of (x1,y1,x2,y2)
    ) -> list[CrossingEvent]:
        """
        Match detections to existing tracks, update positions, detect crossings.
        Returns a list of CrossingEvent for any vehicle that crossed the line this frame.
        """
        events: list[CrossingEvent] = []

        # ── Match detections → tracks via greedy IoU ──────────────────────────
        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        for tid, track in self._tracks.items():
            if not unmatched_dets:
                break
            best_iou   = self.iou_thresh
            best_didx  = -1
            for didx in unmatched_dets:
                score = _iou(track.bbox, detections[didx])
                if score > best_iou:
                    best_iou  = score
                    best_didx = didx
            if best_didx >= 0:
                prev_center = track.center
                track.bbox   = detections[best_didx]
                track.center = _center(detections[best_didx])
                track.hits  += 1
                track.missed = 0
                matched_track_ids.add(tid)
                unmatched_dets.remove(best_didx)

                # ── Check line crossing ───────────────────────────────────────
                if not track.crossed:
                    ev = self._check_crossing(
                        track, prev_center, track.center, frame_rgb
                    )
                    if ev:
                        track.crossed = True
                        events.append(ev)

        # ── Increment missed for unmatched tracks ─────────────────────────────
        dead = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.missed += 1
                if track.missed > self.max_missed:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        # ── Create new tracks for unmatched detections ────────────────────────
        for didx in unmatched_dets:
            bbox   = detections[didx]
            center = _center(bbox)
            tid    = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=bbox,
                center=center,
            )

        return events

    # ── Line-crossing test ────────────────────────────────────────────────────

    def _check_crossing(
        self,
        track:       Track,
        prev_center: tuple[float, float],
        curr_center: tuple[float, float],
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
            crossed = prev_val < lp <= curr_val
        elif self.direction in ("up", "left"):
            crossed = prev_val > lp >= curr_val

        if not crossed:
            return None

        # Crop the vehicle region
        x1, y1, x2, y2 = track.bbox
        h, w = frame_rgb.shape[:2]
        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(w, x2);  y2c = min(h, y2)
        crop = frame_rgb[y1c:y2c, x1c:x2c].copy()

        return CrossingEvent(
            track_id   = track.track_id,
            bbox       = track.bbox,
            frame_crop = crop,
        )

    # ── Active tracks accessor ────────────────────────────────────────────────

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self):
        self._tracks.clear()
        self._next_id = 0


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

LINE_COLOR_IDLE    = (245, 197, 24, 180)   # yellow, semi-transparent
LINE_COLOR_TRIGGER = (255, 59,  59, 220)   # red, brighter when triggered


def draw_virtual_line(
    image_rgb:  np.ndarray,
    line_pos:   int,
    axis:       str  = "horizontal",
    triggered:  bool = False,
    label:      str  = "Detection Line",
) -> np.ndarray:
    """
    Overlay a semi-transparent line on image_rgb (in-place copy returned).
    Works in RGB space.
    """
    overlay = image_rgb.copy()
    h, w    = image_rgb.shape[:2]

    color_bgr = (59, 59, 255) if triggered else (24, 197, 245)  # BGR for cv2
    thickness = 3 if triggered else 2

    if axis == "horizontal":
        y = min(max(line_pos, 0), h - 1)
        # Dashed line effect
        dash_len = 20
        gap_len  = 10
        x = 0
        while x < w:
            x_end = min(x + dash_len, w)
            cv2.line(overlay, (x, y), (x_end, y), color_bgr, thickness)
            x += dash_len + gap_len
        # Label
        label_text = f"▶ {label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (8, y - th - 10), (8 + tw + 10, y - 2), color_bgr, -1)
        cv2.putText(overlay, label_text, (12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        x = min(max(line_pos, 0), w - 1)
        dash_len = 20
        gap_len  = 10
        y_pos = 0
        while y_pos < h:
            y_end = min(y_pos + dash_len, h)
            cv2.line(overlay, (x, y_pos), (x, y_end), color_bgr, thickness)
            y_pos += dash_len + gap_len
        label_text = f"▶ {label}"
        cv2.putText(overlay, label_text, (x + 6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)

    # Blend for semi-transparency
    result = cv2.addWeighted(overlay, 0.85, image_rgb, 0.15, 0)
    return result


def draw_vehicle_tracks(
    image_rgb:   np.ndarray,
    tracks:      list[Track],
    events_this_frame: list[CrossingEvent],
    all_detections_with_belt: dict[int, dict],  # track_id → seatbelt result
) -> np.ndarray:
    """
    Draw bounding boxes for all active tracks:
      • YELLOW  — vehicle detected, not yet crossed line
      • GREEN   — crossed line, seatbelt compliant
      • RED     — crossed line, violation (no seatbelt)
    """
    img = image_rgb.copy()
    crossing_ids = {ev.track_id for ev in events_this_frame}

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        tid = track.track_id

        belt_result = all_detections_with_belt.get(tid)

        if belt_result is None:
            # Not yet inspected
            color = (245, 197, 24)   # yellow
            label = f"ID:{tid}"
        elif belt_result.get("has_belt"):
            color = (0, 229, 160)    # green
            label = f"ID:{tid} ✓ SAFE"
        else:
            color = (255, 59, 59)    # red
            plate = belt_result.get("plate") or "???"
            label = f"ID:{tid} ✗ {plate}"

        lw = 3 if (tid in crossing_ids or belt_result) else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # Show confidence if available
        if belt_result:
            conf_text = f"conf {belt_result.get('confidence', 0):.2f}"
            cv2.putText(img, conf_text, (x1 + 4, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Mock vehicle detector (replace with real YOLO when models are ready)
# ─────────────────────────────────────────────────────────────────────────────

def mock_vehicle_detect(frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Simulate vehicle bounding box detections.
    Replace with:  yolo_model.predict(frame_rgb, classes=[2,3,5,7])  # COCO vehicle classes
    Returns list of (x1, y1, x2, y2).
    """
    import random
    h, w = frame_rgb.shape[:2]
    seed = int(frame_rgb[::40, ::40].mean() * 100) % 9999
    rng  = random.Random(seed)

    n = rng.randint(0, 3)
    bboxes = []
    for _ in range(n):
        x1 = rng.randint(20, w - 200)
        y1 = rng.randint(10, h - 180)
        bw = rng.randint(120, 240)
        bh = rng.randint(100, 200)
        bboxes.append((x1, y1, min(x1+bw, w-1), min(y1+bh, h-1)))
    return bboxes
