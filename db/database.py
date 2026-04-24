# db/database.py
"""
SQLite database layer for MedScribe Rural.
Handles storage and querying of structured patient records.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      TEXT,
                facility        TEXT,
                date            TEXT,
                age             INTEGER,
                sex             TEXT,
                chief_complaint TEXT,
                diagnosis       TEXT,
                icd10_code      TEXT,
                treatment       TEXT,
                outcome         TEXT,
                confidence      REAL,
                flags           TEXT,   -- JSON array
                raw_text        TEXT,
                extracted_at    TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_date        ON records(date);
            CREATE INDEX IF NOT EXISTS idx_icd10       ON records(icd10_code);
            CREATE INDEX IF NOT EXISTS idx_facility    ON records(facility);

            CREATE TABLE IF NOT EXISTS import_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT,
                facility    TEXT,
                n_records   INTEGER,
                imported_at TEXT DEFAULT (datetime('now'))
            );
        """)


def insert_records(records: list[dict]) -> int:
    """Insert a list of structured records. Returns number inserted."""
    if not records:
        return 0

    rows = []
    for r in records:
        rows.append((
            r.get("patient_id"),
            r.get("facility", "Unknown"),
            r.get("date"),
            r.get("age"),
            r.get("sex"),
            r.get("chief_complaint"),
            r.get("diagnosis"),
            r.get("icd10_code"),
            r.get("treatment"),
            r.get("outcome"),
            r.get("confidence"),
            json.dumps(r.get("flags", [])),
            r.get("raw_text"),
            r.get("extracted_at"),
        ))

    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO records
            (patient_id, facility, date, age, sex, chief_complaint,
             diagnosis, icd10_code, treatment, outcome, confidence,
             flags, raw_text, extracted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    return len(rows)


def log_import(filename: str, facility: str, n_records: int):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO import_log (filename, facility, n_records) VALUES (?,?,?)",
            (filename, facility, n_records)
        )


def get_records(facility: str = None, days: int = 30) -> list[dict]:
    """Fetch records from the last N days, optionally filtered by facility."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = "SELECT * FROM records WHERE date >= ?"
    params = [since]
    if facility:
        query += " AND facility = ?"
        params.append(facility)
    query += " ORDER BY date DESC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_disease_summary(days: int = 7) -> list[dict]:
    """Aggregate case counts by ICD-10 code for the last N days."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT icd10_code, diagnosis, COUNT(*) as case_count, facility
            FROM records
            WHERE date >= ?
            GROUP BY icd10_code, facility
            ORDER BY case_count DESC
        """, [since]).fetchall()
    return [dict(r) for r in rows]


def get_flagged_records() -> list[dict]:
    """Return records with low confidence or flags for manual review."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM records
            WHERE confidence < 0.75 OR flags != '[]'
            ORDER BY created_at DESC
            LIMIT 100
        """).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Return high-level database statistics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        facilities = conn.execute("SELECT COUNT(DISTINCT facility) FROM records").fetchone()[0]
        flagged = conn.execute("SELECT COUNT(*) FROM records WHERE confidence < 0.75").fetchone()[0]
        last_import = conn.execute("SELECT MAX(imported_at) FROM import_log").fetchone()[0]
    return {
        "total_records": total,
        "facilities": facilities,
        "flagged_records": flagged,
        "last_import": last_import or "Never"
    }
