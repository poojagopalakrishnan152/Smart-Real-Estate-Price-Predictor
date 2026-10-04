# database.py
# SQLite Database Handler for Smart Real Estate Price Predictor – Bengaluru
# Manages prediction history storage and retrieval.
# STRICTLY NO bedroom, bathroom, BHK, or balcony fields are stored.

import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "predictions.db"


def get_connection():
    """Returns a SQLite database connection with row factory enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location TEXT NOT NULL,
            total_sqft REAL NOT NULL,
            area_type TEXT NOT NULL,
            availability TEXT NOT NULL,
            society_status TEXT NOT NULL,
            predicted_price_lakhs REAL NOT NULL,
            formatted_price TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction_record(location, total_sqft, area_type, availability, society_status, price_lakhs, formatted_price):
    """
    Inserts a newly calculated real estate prediction record into SQLite history.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO prediction_history (
            timestamp, location, total_sqft, area_type,
            availability, society_status, predicted_price_lakhs, formatted_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        str(location).strip(),
        float(total_sqft),
        str(area_type).strip(),
        str(availability).strip(),
        str(society_status).strip(),
        float(price_lakhs),
        str(formatted_price).strip()
    ))
    conn.commit()
    conn.close()


def fetch_prediction_history(limit=100):
    """
    Fetches the latest prediction records ordered by most recent first.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, location, total_sqft, area_type,
               availability, society_status, predicted_price_lakhs, formatted_price
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_all_predictions():
    """Clears all records from the prediction history table."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at: {DB_PATH}")
