import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import psycopg2
import pandas as pd
from datetime import datetime
import io
import base64

# ─── Line Crossing Module ─────────────────────────────────────────────────────
from line_crossing import (
    LineCrossingTracker,
    draw_virtual_line,
    draw_vehicle_tracks,
    mock_vehicle_detect,
    CrossingEvent,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeRoad AI — Violation Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0b0d12;
    --surface: #12151e;
    --card: #181c28;
    --border: #252a3a;
    --accent: #f5c518;
    --danger: #ff3b3b;
    --safe: #00e5a0;
    --muted: #5a6075;
    --text: #e8eaf2;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

.stApp {
    background-color: var(--bg);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.yellow::before { background: var(--accent); }
.metric-card.red::before    { background: var(--danger); }
.metric-card.green::before  { background: var(--safe); }

.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
}

/* Violation badge */
.violation-badge {
    display: inline-block;
    background: rgba(255,59,59,0.15);
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    font-weight: 700;
}
.safe-badge {
    display: inline-block;
    background: rgba(0,229,160,0.1);
    color: var(--safe);
    border: 1px solid var(--safe);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
}

/* Section title */
.section-title {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
}

/* Plate display */
.plate-display {
    background: #1a1a00;
    border: 2px solid var(--accent);
    border-radius: 8px;
    padding: 10px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.15em;
    display: inline-block;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0b0d12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* Tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* Progress */
.stProgress > div > div {
    background: var(--accent) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Log entry */
.log-entry {
    background: var(--card);
    border-left: 3px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.log-entry.safe { border-left-color: var(--safe); }

/* Top bar */
.top-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ─── Database Helper ──────────────────────────────────────────────────────────
def get_db_connection():
    """Connect to PostgreSQL. Returns None if not configured."""
    try:
        db_url = st.session_state.get("db_url", "")
        if not db_url:
            return None
        conn = psycopg2.connect(db_url, connect_timeout=5)
        return conn
    except Exception:
        return None


def init_db(conn):
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id SERIAL PRIMARY KEY,
                plate_number TEXT,
                violation_type TEXT DEFAULT 'No Seatbelt',
                confidence REAL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                image_name TEXT
            );
        """)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"DB init error: {e}")
        return False


def insert_violation(conn, plate_number, violation_type, confidence, image_name):
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO violations (plate_number, violation_type, confidence, image_name) VALUES (%s,%s,%s,%s)",
            (plate_number, violation_type, float(confidence), image_name)
        )
        conn.commit()
        return True
    except Exception:
        return False


def fetch_violations(conn):
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql("SELECT * FROM violations ORDER BY timestamp DESC LIMIT 200", conn)
        return df
    except Exception:
        return pd.DataFrame()


# ─── Mock Detection (replace with real YOLO/OCR) ─────────────────────────────
def mock_detect(image_np):
    """
    Simulates YOLO + OCR pipeline.
    Replace this function with actual model inference.
    Returns list of detection dicts.
    """
    h, w = image_np.shape[:2]
    import random
    random.seed(int(image_np.mean()))

    detections = []
    n = random.randint(1, 3)
    for i in range(n):
        has_belt = random.random() > 0.45
        x1 = random.randint(50, w // 2)
        y1 = random.randint(30, h // 2)
        x2 = x1 + random.randint(80, 180)
        y2 = y1 + random.randint(100, 220)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        plate = None
        if not has_belt:
            letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
            digits  = "0123456789"
            plate = (
                random.choice(letters) + random.choice(letters) +
                "-" + str(random.randint(10, 99)) +
                "-" + random.choice(letters) + random.choice(letters) + random.choice(letters)
            )

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "has_belt": has_belt,
            "confidence": round(random.uniform(0.72, 0.98), 2),
            "plate": plate
        })
    return detections


def draw_detections(image_np, detections):
    img = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 229, 160) if det["has_belt"] else (59, 59, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = "With Belt" if det["has_belt"] else f"VIOLATION"
        cv2.rectangle(img, (x1, y1 - 22), (x1 + len(label) * 11 + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        if det["plate"]:
            cv2.putText(img, det["plate"], (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 197, 24), 1, cv2.LINE_AA)
    return img


# ─── Session State Init ───────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []  # list of result dicts
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "db_url" not in st.session_state:
    st.session_state.db_url = ""
# Line-crossing specific state
if "lc_tracker" not in st.session_state:
    st.session_state.lc_tracker = None
if "lc_results" not in st.session_state:
    st.session_state.lc_results = {}   # track_id → seatbelt/plate dict
if "lc_events_log" not in st.session_state:
    st.session_state.lc_events_log = []  # list of crossing event dicts


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:24px'>
        <div style='font-size:1.5rem;font-weight:800;color:#f5c518;letter-spacing:-0.02em'>🚦 SafeRoad AI</div>
        <div style='font-size:0.7rem;color:#5a6075;letter-spacing:0.12em;text-transform:uppercase;font-family:"Space Mono",monospace'>Violation Monitor v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Database Config</div>', unsafe_allow_html=True)
    db_url = st.text_input(
        "PostgreSQL URL",
        value=st.session_state.db_url,
        type="password",
        placeholder="postgresql://user:pass@host:5432/db",
        help="Leave blank to use in-memory logging only"
    )
    if db_url != st.session_state.db_url:
        st.session_state.db_url = db_url

    conn = get_db_connection()
    if conn:
        if init_db(conn):
            st.success("✓ Database connected")
    else:
        st.caption("⚠ No DB — results stored in session only")

    st.divider()
    st.markdown('<div class="section-title">Virtual Line Settings</div>', unsafe_allow_html=True)
    lc_axis      = st.selectbox("Line Axis", ["horizontal", "vertical"], index=0)
    lc_direction = st.selectbox(
        "Crossing Direction",
        ["down", "up"] if lc_axis == "horizontal" else ["right", "left"],
        index=0
    )
    lc_label = st.text_input("Line Label", value="Detection Zone")

    st.divider()
    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.3, 1.0, 0.6, 0.05)
    iou_thresh  = st.slider("IoU Threshold (NMS)", 0.3, 1.0, 0.45, 0.05)
    st.caption("Model: YOLOv8 · OCR: Tesseract")

    st.divider()
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.caption(
        "This system detects seatbelt violations in images/video, "
        "extracts license plate numbers via OCR, and logs all "
        "violations to PostgreSQL."
    )


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:flex-end;gap:16px;margin-bottom:8px'>
    <div>
        <div style='font-size:2.2rem;font-weight:800;line-height:1;color:#e8eaf2'>Seatbelt Violation</div>
        <div style='font-size:2.2rem;font-weight:800;line-height:1;color:#f5c518'>Detection System</div>
    </div>
    <div style='margin-bottom:6px;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;
                color:#5a6075;font-family:"Space Mono",monospace;padding-bottom:4px'>
        YOLO · OCR · PostgreSQL · Streamlit
    </div>
</div>
<div style='height:1px;background:linear-gradient(90deg,#f5c518,transparent);margin-bottom:28px'></div>
""", unsafe_allow_html=True)


# ─── KPI Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
violations_today = st.session_state.total_violations
processed_today  = st.session_state.total_processed
vrate = f"{(violations_today/processed_today*100):.0f}%" if processed_today else "—"

with col1:
    st.markdown(f"""
    <div class="metric-card yellow">
        <div class="metric-value" style="color:#f5c518">{processed_today}</div>
        <div class="metric-label">Frames Processed</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card red">
        <div class="metric-value" style="color:#ff3b3b">{violations_today}</div>
        <div class="metric-label">Violations Found</div>
    </div>""", unsafe_allow_html=True)
with col3:
    safe_count = processed_today - violations_today
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-value" style="color:#00e5a0">{safe_count}</div>
        <div class="metric-label">Compliant</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card yellow">
        <div class="metric-value" style="color:#f5c518">{vrate}</div>
        <div class="metric-label">Violation Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)


# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📷  Image Detection", "🎬  Video Detection", "🗃  Violation Log", "📏  Line Crossing"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["jpg","jpeg","png","bmp","webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_np = np.array(img)
            st.image(img, caption="Input Image", use_container_width=True)

            if st.button("🔍  Run Detection", use_container_width=True):
                with st.spinner("Running YOLO + OCR pipeline..."):
                    progress = st.progress(0)
                    time.sleep(0.3); progress.progress(25, "Detecting persons…")
                    time.sleep(0.3); progress.progress(55, "Detecting license plates…")
                    detections = mock_detect(img_np)
                    time.sleep(0.3); progress.progress(80, "Applying OCR…")
                    annotated  = draw_detections(img_np, detections)
                    time.sleep(0.2); progress.progress(100, "Done ✓")

                # Save results to session & DB
                for det in detections:
                    st.session_state.total_processed += 1
                    if not det["has_belt"]:
                        st.session_state.total_violations += 1
                        entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "plate": det["plate"] or "UNREADABLE",
                            "violation": "No Seatbelt",
                            "confidence": det["confidence"],
                            "source": uploaded.name
                        }
                        st.session_state.log.insert(0, entry)
                        if conn:
                            insert_violation(conn, entry["plate"], entry["violation"],
                                             entry["confidence"], uploaded.name)

                st.session_state["last_annotated"] = annotated
                st.session_state["last_detections"] = detections
                st.rerun()

    with col_right:
        st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)

        if "last_annotated" in st.session_state:
            st.image(st.session_state["last_annotated"], caption="Annotated Output",
                     use_container_width=True)
            dets = st.session_state["last_detections"]
            total   = len(dets)
            viols   = sum(1 for d in dets if not d["has_belt"])
            safe_n  = total - viols

            st.markdown(f"""
            <div style='display:flex;gap:12px;margin-top:12px;flex-wrap:wrap'>
                <div class='metric-card yellow' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#f5c518;font-size:1.6rem'>{total}</div>
                    <div class='metric-label'>Detected</div>
                </div>
                <div class='metric-card red' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#ff3b3b;font-size:1.6rem'>{viols}</div>
                    <div class='metric-label'>Violations</div>
                </div>
                <div class='metric-card green' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#00e5a0;font-size:1.6rem'>{safe_n}</div>
                    <div class='metric-label'>Safe</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            for i, det in enumerate(dets):
                if not det["has_belt"]:
                    st.markdown(f"""
                    <div class="log-entry">
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <span class='violation-badge'>⚠ VIOLATION</span>
                            <span style='color:#5a6075'>conf: {det['confidence']}</span>
                        </div>
                        <div style='margin-top:8px'>
                            Plate: <span class='plate-display'>{det['plate'] or 'UNREADABLE'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="log-entry safe">
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <span class='safe-badge'>✓ COMPLIANT</span>
                            <span style='color:#5a6075'>conf: {det['confidence']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                        padding:60px;text-align:center;color:#5a6075'>
                <div style='font-size:2.5rem;margin-bottom:12px'>📷</div>
                <div style='font-family:"Space Mono",monospace;font-size:0.8rem'>
                    Upload an image and run detection
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Video Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_v1, col_v2 = st.columns([1, 1], gap="large")

    with col_v1:
        st.markdown('<div class="section-title">Upload Video</div>', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Drop a video file",
            type=["mp4","avi","mov","mkv"],
            label_visibility="collapsed",
            key="video_uploader"
        )
        if video_file:
            st.video(video_file)

            sample_every = st.number_input("Process every N-th frame", 1, 60, 10)

            if st.button("▶  Process Video", use_container_width=True):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30

                progress_bar = st.progress(0)
                status_text  = st.empty()
                frame_num    = 0
                processed    = 0
                video_viols  = 0

                preview_slot = col_v2.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_num % sample_every == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        dets = mock_detect(rgb)
                        annotated = draw_detections(rgb, dets)
                        preview_slot.image(annotated, caption=f"Frame {frame_num}", use_container_width=True)

                        for det in dets:
                            st.session_state.total_processed += 1
                            processed += 1
                            if not det["has_belt"]:
                                st.session_state.total_violations += 1
                                video_viols += 1
                                ts = f"{int(frame_num/fps//60):02d}:{int(frame_num/fps%60):02d}"
                                entry = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "plate": det["plate"] or "UNREADABLE",
                                    "violation": "No Seatbelt",
                                    "confidence": det["confidence"],
                                    "source": f"{video_file.name} @ {ts}"
                                }
                                st.session_state.log.insert(0, entry)
                                if conn:
                                    insert_violation(conn, entry["plate"], entry["violation"],
                                                     entry["confidence"], entry["source"])

                    pct = min(int(frame_num / max(total_frames, 1) * 100), 100)
                    progress_bar.progress(pct)
                    status_text.caption(f"Frame {frame_num}/{total_frames} · {processed} sampled · {video_viols} violations")
                    frame_num += 1

                cap.release()
                os.unlink(tmp_path)
                status_text.success(f"✓ Processed {processed} frames — {video_viols} violations logged")

    with col_v2:
        st.markdown('<div class="section-title">Live Preview</div>', unsafe_allow_html=True)
        if "last_annotated" not in st.session_state:
            st.markdown("""
            <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                        padding:60px;text-align:center;color:#5a6075'>
                <div style='font-size:2.5rem;margin-bottom:12px'>🎬</div>
                <div style='font-family:"Space Mono",monospace;font-size:0.8rem'>
                    Upload a video and press Process
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Violation Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Violation Log</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🔄  Refresh", use_container_width=True):
            st.rerun()
        if st.button("🗑  Clear Session Log", use_container_width=True):
            st.session_state.log = []
            st.session_state.total_processed = 0
            st.session_state.total_violations = 0
            st.rerun()

    # Pull from DB if available, else use session log
    db_df = fetch_violations(conn)

    if not db_df.empty:
        st.caption(f"Showing {len(db_df)} records from PostgreSQL")
        st.dataframe(
            db_df[["id","plate_number","violation_type","confidence","timestamp","image_name"]].rename(columns={
                "plate_number": "Plate",
                "violation_type": "Violation",
                "confidence": "Conf.",
                "timestamp": "Timestamp",
                "image_name": "Source"
            }),
            use_container_width=True,
            hide_index=True
        )
    elif st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.caption(f"Showing {len(df)} records from session (no DB connected)")
        st.dataframe(
            df.rename(columns={
                "timestamp": "Timestamp",
                "plate": "Plate",
                "violation": "Violation",
                "confidence": "Conf.",
                "source": "Source"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Export CSV",
            csv,
            file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown("""
        <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                    padding:48px;text-align:center;color:#5a6075;margin-top:16px'>
            <div style='font-size:2rem;margin-bottom:10px'>🗃</div>
            <div style='font-family:"Space Mono",monospace;font-size:0.8rem'>
                No violations logged yet. Run detection on an image or video.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recent log entries (session)
    if st.session_state.log:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)
        for entry in st.session_state.log[:10]:
            st.markdown(f"""
            <div class="log-entry">
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                    <span class='violation-badge'>⚠ {entry['violation']}</span>
                    <span style='color:#5a6075;font-size:0.72rem'>{entry['timestamp']}</span>
                </div>
                <div>Plate: <span style='color:#f5c518;font-weight:700;letter-spacing:0.1em'>{entry['plate']}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Conf: <span style='color:#e8eaf2'>{entry['confidence']}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Source: <span style='color:#5a6075'>{entry['source']}</span></div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Virtual Line Crossing Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("""
    <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                padding:16px 20px;margin-bottom:20px'>
        <div style='font-size:0.75rem;color:#5a6075;font-family:"Space Mono",monospace;
                    line-height:1.7'>
            <span style='color:#f5c518;font-weight:700'>How it works:</span>
            Define a virtual detection line → Upload video → Vehicles that cross the line are
            automatically inspected for seatbelt compliance → Violations are logged with plate OCR.
        </div>
    </div>
    """, unsafe_allow_html=True)

    lc_col1, lc_col2 = st.columns([1, 1], gap="large")

    # ── Left Column: Config + Upload ──────────────────────────────────────────
    with lc_col1:
        st.markdown('<div class="section-title">⚙ Line Configuration</div>', unsafe_allow_html=True)

        lc_video_file = st.file_uploader(
            "Upload video for line-crossing analysis",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            key="lc_video_uploader"
        )

        # Preview first frame so user can place the line
        if lc_video_file:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_lc:
                tmp_lc.write(lc_video_file.read())
                lc_tmp_path = tmp_lc.name

            cap_preview = cv2.VideoCapture(lc_tmp_path)
            ret_p, first_frame = cap_preview.read()
            cap_preview.release()

            if ret_p:
                first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                h_vid, w_vid = first_rgb.shape[:2]

                # Line position slider — range depends on axis
                if lc_axis == "horizontal":
                    lc_line_pos = st.slider(
                        "Line Position (Y pixel)",
                        min_value=20, max_value=h_vid - 20,
                        value=h_vid // 2, step=2,
                        help="Drag to place the horizontal detection line"
                    )
                else:
                    lc_line_pos = st.slider(
                        "Line Position (X pixel)",
                        min_value=20, max_value=w_vid - 20,
                        value=w_vid // 2, step=2,
                        help="Drag to place the vertical detection line"
                    )

                # Draw line on first frame preview
                preview_with_line = draw_virtual_line(
                    first_rgb, lc_line_pos, lc_axis,
                    triggered=False, label=lc_label
                )
                st.image(preview_with_line,
                         caption=f"Line preview — {lc_axis} at {'Y' if lc_axis=='horizontal' else 'X'}={lc_line_pos}",
                         use_container_width=True)

                # Processing options
                st.markdown('<div class="section-title">Processing Options</div>', unsafe_allow_html=True)
                lc_sample_every = st.number_input("Process every N-th frame", 1, 30, 3, key="lc_sample")
                lc_iou_thresh   = st.slider("Tracker IoU threshold", 0.15, 0.7, 0.3, 0.05, key="lc_iou")
                lc_max_missed   = st.number_input("Max missed frames before drop", 2, 30, 8, key="lc_missed")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    run_lc = st.button("▶  Start Analysis", use_container_width=True, key="run_lc")
                with col_btn2:
                    reset_lc = st.button("↺  Reset Tracker", use_container_width=True, key="reset_lc")

                if reset_lc:
                    st.session_state.lc_tracker  = None
                    st.session_state.lc_results  = {}
                    st.session_state.lc_events_log = []
                    st.success("Tracker reset.")

                # ── Run the analysis ──────────────────────────────────────────
                if run_lc:
                    tracker = LineCrossingTracker(
                        line_pos   = lc_line_pos,
                        direction  = lc_direction,
                        axis       = lc_axis,
                        iou_thresh = lc_iou_thresh,
                        max_missed = lc_max_missed,
                    )
                    st.session_state.lc_tracker = tracker
                    st.session_state.lc_results = {}
                    st.session_state.lc_events_log = []

                    cap_lc     = cv2.VideoCapture(lc_tmp_path)
                    total_lc   = int(cap_lc.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_lc     = cap_lc.get(cv2.CAP_PROP_FPS) or 30

                    prog_lc    = st.progress(0)
                    stat_lc    = st.empty()
                    preview_lc = lc_col2.empty()

                    frame_num_lc   = 0
                    crossing_count = 0
                    viol_count_lc  = 0

                    while cap_lc.isOpened():
                        ret_lc, frame_lc = cap_lc.read()
                        if not ret_lc:
                            break

                        if frame_num_lc % lc_sample_every == 0:
                            rgb_lc = cv2.cvtColor(frame_lc, cv2.COLOR_BGR2RGB)

                            # ── Step 1: Detect vehicles in this frame ─────────
                            # SWAP mock_vehicle_detect → real YOLO vehicle model:
                            # vehicle_bboxes = vehicle_yolo.predict(rgb_lc, classes=[2,3,5,7])[0].boxes.xyxy.int().tolist()
                            vehicle_bboxes = mock_vehicle_detect(rgb_lc)

                            # ── Step 2: Update tracker, get crossing events ───
                            events = tracker.update(rgb_lc, vehicle_bboxes)

                            # ── Step 3: For each crossing event, inspect the crop ──
                            for ev in events:
                                crossing_count += 1

                                # Seatbelt detection on cropped vehicle region
                                # SWAP: real_detections = seatbelt_detector.detect(ev.frame_crop)
                                crop_dets = mock_detect(ev.frame_crop)

                                has_belt    = all(d["has_belt"] for d in crop_dets) if crop_dets else True
                                top_det     = crop_dets[0] if crop_dets else {}
                                confidence  = top_det.get("confidence", 0.0)
                                plate_num   = top_det.get("plate") if not has_belt else None

                                result = {
                                    "has_belt":   has_belt,
                                    "confidence": confidence,
                                    "plate":      plate_num,
                                    "frame":      frame_num_lc,
                                    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "bbox":       ev.bbox,
                                }
                                st.session_state.lc_results[ev.track_id] = result

                                if not has_belt:
                                    viol_count_lc += 1
                                    entry = {
                                        "timestamp": result["timestamp"],
                                        "plate":     plate_num or "UNREADABLE",
                                        "violation": "No Seatbelt (Line Crossing)",
                                        "confidence": confidence,
                                        "source":    f"Frame {frame_num_lc} — Line @ {lc_line_pos}",
                                        "track_id":  ev.track_id,
                                    }
                                    st.session_state.lc_events_log.insert(0, entry)
                                    st.session_state.log.insert(0, entry)
                                    st.session_state.total_violations += 1

                                    if conn:
                                        insert_violation(
                                            conn,
                                            entry["plate"],
                                            "No Seatbelt",
                                            confidence,
                                            entry["source"],
                                        )

                                st.session_state.total_processed += 1

                            # ── Step 4: Draw everything ───────────────────────
                            frame_vis = draw_virtual_line(
                                rgb_lc, lc_line_pos, lc_axis,
                                triggered=len(events) > 0,
                                label=lc_label,
                            )
                            frame_vis = draw_vehicle_tracks(
                                frame_vis,
                                tracker.tracks,
                                events,
                                st.session_state.lc_results,
                            )

                            preview_lc.image(
                                frame_vis,
                                caption=f"Frame {frame_num_lc} | Crossings: {crossing_count} | Violations: {viol_count_lc}",
                                use_container_width=True,
                            )

                        pct_lc = min(int(frame_num_lc / max(total_lc, 1) * 100), 100)
                        prog_lc.progress(pct_lc)
                        stat_lc.caption(
                            f"Frame {frame_num_lc}/{total_lc} · "
                            f"Crossings: {crossing_count} · "
                            f"Violations: {viol_count_lc}"
                        )
                        frame_num_lc += 1

                    cap_lc.release()
                    os.unlink(lc_tmp_path)
                    stat_lc.success(
                        f"✓ Done — {crossing_count} vehicles crossed the line, "
                        f"{viol_count_lc} violations logged."
                    )

            else:
                st.error("Could not read video file.")

        else:
            st.markdown("""
            <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                        padding:48px;text-align:center;color:#5a6075'>
                <div style='font-size:2.5rem;margin-bottom:12px'>📏</div>
                <div style='font-family:"Space Mono",monospace;font-size:0.8rem'>
                    Upload a video to configure the detection line
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right Column: Live preview + Event Log ────────────────────────────────
    with lc_col2:
        st.markdown('<div class="section-title">Live Preview</div>', unsafe_allow_html=True)

        if not lc_video_file:
            st.markdown("""
            <div style='background:#12151e;border:1px solid #252a3a;border-radius:12px;
                        padding:80px;text-align:center;color:#5a6075'>
                <div style='font-size:2.5rem;margin-bottom:12px'>🎯</div>
                <div style='font-family:"Space Mono",monospace;font-size:0.8rem'>
                    Annotated video will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Crossing Events Log ───────────────────────────────────────────────
        if st.session_state.lc_events_log:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Crossing Events</div>', unsafe_allow_html=True)

            # Summary mini-metrics
            total_ev   = len(st.session_state.lc_events_log)
            viol_ev    = sum(1 for e in st.session_state.lc_events_log if "No Seatbelt" in e["violation"])
            safe_ev    = total_ev - viol_ev

            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap'>
                <div class='metric-card yellow' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#f5c518;font-size:1.5rem'>{total_ev}</div>
                    <div class='metric-label'>Total Crossings</div>
                </div>
                <div class='metric-card red' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#ff3b3b;font-size:1.5rem'>{viol_ev}</div>
                    <div class='metric-label'>Violations</div>
                </div>
                <div class='metric-card green' style='flex:1;min-width:80px;padding:12px'>
                    <div class='metric-value' style='color:#00e5a0;font-size:1.5rem'>{safe_ev}</div>
                    <div class='metric-label'>Compliant</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for ev in st.session_state.lc_events_log[:15]:
                is_viol = "No Seatbelt" in ev["violation"]
                border  = "#ff3b3b" if is_viol else "#00e5a0"
                badge   = f"<span class='violation-badge'>⚠ VIOLATION</span>" if is_viol else f"<span class='safe-badge'>✓ COMPLIANT</span>"
                plate_html = f"Plate: <span style='color:#f5c518;font-weight:700;letter-spacing:0.1em'>{ev['plate']}</span> &nbsp;|&nbsp;" if is_viol else ""
                st.markdown(f"""
                <div class="log-entry" style='border-left-color:{border}'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                        {badge}
                        <span style='color:#5a6075;font-size:0.72rem'>ID:{ev.get("track_id","?")} · {ev['timestamp']}</span>
                    </div>
                    <div style='font-size:0.78rem'>
                        {plate_html}Conf: <span style='color:#e8eaf2'>{ev['confidence']}</span>
                        &nbsp;|&nbsp; <span style='color:#5a6075'>{ev['source']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Export crossing events as CSV
            lc_df = pd.DataFrame(st.session_state.lc_events_log)
            lc_csv = lc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Export Crossing Events CSV",
                lc_csv,
                file_name=f"crossings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
