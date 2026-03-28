import sqlite3
from pathlib import Path
from datetime import datetime
import json
'''
Logging every eval run to sqlite to analyze later
'''

DB_PATH = Path("data/eval_results.db")

def _conn() -> sqlite3.Connection: # standard pattern for getting a connection, ensuring the directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    with _conn() as conn: # create tables if they don't exist
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
        conn.execute( # store individual question results for more detailed analysis later
            """
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

def save_run(run_id: str, scores: dict, samples: list[dict], notes: str = "") -> None: # store a new eval run
    init_db()
    ran_at = datetime.utcnow().isoformat() # store the time of the run in ISO format for easy sorting and analysis
    with _conn() as conn:
        conn.execute("""
            INSERT INTO eval_runs
                (run_id, ran_at, num_questions, faithfulness,
                 answer_relevancy, context_recall, context_precision, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, ran_at, scores.get("num_questions"),
            scores.get("faithfulness"), scores.get("answer_relevancy"),
            scores.get("context_recall"), scores.get("context_precision"),
            notes,
        ))
        for s in samples:
            conn.execute("""
                INSERT INTO eval_samples
                    (run_id, question, ground_truth, answer, contexts,
                     faithfulness, answer_relevancy, context_recall, context_precision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, s.get("question"), s.get("ground_truth"),
                s.get("answer"), json.dumps(s.get("contexts", [])),
                s.get("faithfulness"), s.get("answer_relevancy"),
                s.get("context_recall"), s.get("context_precision"),
            ))
            
def get_runs() -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT run_id, ran_at, num_questions, faithfulness,
                   answer_relevancy, context_recall, context_precision, notes
            FROM eval_runs ORDER BY ran_at DESC
        """).fetchall()
    keys = ["run_id", "ran_at", "num_questions", "faithfulness",
            "answer_relevancy", "context_recall", "context_precision", "notes"]
    return [dict(zip(keys, r)) for r in rows]