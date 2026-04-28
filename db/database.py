"""
db/database.py
──────────────
PostgreSQL integration for violation logging.
"""

from __future__ import annotations
import psycopg2
import pandas as pd
from datetime import datetime
from typing import Optional


DDL = """
CREATE TABLE IF NOT EXISTS violations (
    id             SERIAL PRIMARY KEY,
    plate_number   TEXT,
    violation_type TEXT        DEFAULT 'No Seatbelt',
    confidence     REAL,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    image_name     TEXT,
    frame_number   INT
);

CREATE INDEX IF NOT EXISTS idx_violations_plate     ON violations(plate_number);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
"""


class ViolationDB:
    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url, connect_timeout=10)
        self.conn.autocommit = False
        self._init_schema()

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(DDL)
        self.conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────
    def log_violation(
        self,
        plate_number:   str,
        violation_type: str   = "No Seatbelt",
        confidence:     float = 0.0,
        image_name:     str   = "",
        frame_number:   Optional[int] = None,
    ) -> int:
        """Insert a violation row and return its id."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO violations
                    (plate_number, violation_type, confidence, image_name, frame_number)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (plate_number, violation_type, float(confidence), image_name, frame_number)
            )
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    # ── Read ──────────────────────────────────────────────────────────────────
    def fetch_recent(self, limit: int = 200) -> pd.DataFrame:
        return pd.read_sql(
            f"SELECT * FROM violations ORDER BY timestamp DESC LIMIT {limit}",
            self.conn
        )

    def fetch_by_plate(self, plate: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM violations WHERE plate_number = %s ORDER BY timestamp DESC",
            self.conn, params=(plate,)
        )

    def summary_stats(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                          AS total,
                    COUNT(DISTINCT plate_number)                      AS unique_plates,
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 day') AS today
                FROM violations
            """)
            total, unique, today = cur.fetchone()
        return {"total": total, "unique_plates": unique, "today": today}

    def close(self):
        self.conn.close()
