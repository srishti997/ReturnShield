from pathlib import Path
import sqlite3
from datetime import datetime


# =========================================================
# DATABASE LOCATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = (
    DATA_DIR
    / "returnshield_live.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    # -----------------------------------------------------
    # LIVE CUSTOMERS
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS live_customers (
            customer_id TEXT PRIMARY KEY,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            total_return_requests INTEGER NOT NULL DEFAULT 0,
            total_return_value REAL NOT NULL DEFAULT 0
        )
        """
    )

    # -----------------------------------------------------
    # RETURN EVENTS
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS live_return_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            order_value REAL NOT NULL,
            return_reason TEXT NOT NULL,
            days_to_return INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL,

            FOREIGN KEY (customer_id)
                REFERENCES live_customers(customer_id)
        )
        """
    )

    connection.commit()

    connection.close()


# =========================================================
# CHECK WHETHER LIVE CUSTOMER EXISTS
# =========================================================

def get_live_customer(
    customer_id,
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM live_customers
        WHERE customer_id = ?
        """,
        (
            customer_id,
        ),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:

        return None

    return dict(
        row
    )


# =========================================================
# CREATE LIVE CUSTOMER
# =========================================================

def create_live_customer(
    customer_id,
):

    now = datetime.now().isoformat()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO live_customers (
            customer_id,
            first_seen_at,
            last_seen_at,
            total_return_requests,
            total_return_value
        )
        VALUES (?, ?, ?, 0, 0)
        """,
        (
            customer_id,
            now,
            now,
        ),
    )

    connection.commit()

    connection.close()


# =========================================================
# SAVE RETURN EVENT
# =========================================================

def save_return_event(
    customer_id,
    order_value,
    return_reason,
    days_to_return,
    risk_score,
    risk_level,
):

    now = datetime.now().isoformat()

    connection = get_connection()

    cursor = connection.cursor()

    # -----------------------------------------------------
    # MAKE SURE CUSTOMER EXISTS
    # -----------------------------------------------------

    cursor.execute(
        """
        INSERT OR IGNORE INTO live_customers (
            customer_id,
            first_seen_at,
            last_seen_at,
            total_return_requests,
            total_return_value
        )
        VALUES (?, ?, ?, 0, 0)
        """,
        (
            customer_id,
            now,
            now,
        ),
    )

    # -----------------------------------------------------
    # SAVE RETURN
    # -----------------------------------------------------

    cursor.execute(
        """
        INSERT INTO live_return_events (
            customer_id,
            order_value,
            return_reason,
            days_to_return,
            risk_score,
            risk_level,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            float(
                order_value
            ),
            return_reason,
            int(
                days_to_return
            ),
            float(
                risk_score
            ),
            risk_level,
            now,
        ),
    )

    # -----------------------------------------------------
    # UPDATE CUSTOMER HISTORY
    # -----------------------------------------------------

    cursor.execute(
        """
        UPDATE live_customers

        SET
            last_seen_at = ?,
            total_return_requests =
                total_return_requests + 1,
            total_return_value =
                total_return_value + ?

        WHERE customer_id = ?
        """,
        (
            now,
            float(
                order_value
            ),
            customer_id,
        ),
    )

    connection.commit()

    connection.close()


# =========================================================
# GET CUSTOMER RETURN HISTORY
# =========================================================

def get_customer_return_history(
    customer_id,
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM live_return_events

        WHERE customer_id = ?

        ORDER BY created_at DESC
        """,
        (
            customer_id,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# =========================================================
# INITIALIZE WHEN RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    initialize_database()

    print(
        "ReturnShield live database initialized."
    )

    print(
        f"Database location: {DATABASE_PATH}"
    )