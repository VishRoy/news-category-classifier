import sqlite3
from datetime import datetime

DB_PATH = 'predictions.db'


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                headline TEXT,
                predicted TEXT,
                confidence REAL,
                model_version TEXT,
                is_correct INTEGER,
                correct_label TEXT
            )
        """)


def log_prediction(headline, predicted, confidence, model_version):
    with sqlite3.connect(DB_PATH) as conn:
        curr = conn.execute(
            "INSERT INTO predictions (created_at, headline, predicted, confidence, model_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"),
             headline, predicted, float(confidence), model_version),
        )
        return curr.lastrowid


def save_feedback(row_id, is_correct, correct_label):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE predictions SET is_correct = ?, correct_label = ? WHERE id = ?",
            (int(is_correct), correct_label, row_id),
        )


def get_prediction(row_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            'SELECT predicted FROM predictions WHERE id = ?', (row_id,)
        ).fetchone()
    return row[0] if row else None
