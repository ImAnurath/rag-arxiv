import sqlite3
from pathlib import Path
from datetime import datetime
import json
'''
Logging every eval run to sqlite to analyze later
'''

DB_PATH = Path("data/eval_results.db")

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                ran_at          TEXT NOT NULL,
                num_questions   INTEGER,
                faithfulness    REAL,
                answer_relevancy REAL,
                context_recall  REAL,
                context_precision REAL,
                notes           TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_samples (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                question        TEXT,
                ground_truth    TEXT,
                answer          TEXT,
                contexts        TEXT,
                faithfulness    REAL,
                answer_relevancy REAL,
                context_recall  REAL,
                context_precision REAL
            )
        """)