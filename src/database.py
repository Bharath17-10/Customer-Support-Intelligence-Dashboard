import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

STATUSES = ("Pending", "In Progress", "Resolved")
CUSTOMER_RECORD = "Customer"
DEMO_RECORD = "Demo / Dataset Imported"
DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "support_tickets.db"
)


def _connect(database_path=DEFAULT_DATABASE_PATH):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=10)


@contextmanager
def _database_connection(database_path):
    connection = _connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(database_path=DEFAULT_DATABASE_PATH):
    with _database_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                ticket_text TEXT NOT NULL,
                department TEXT NOT NULL,
                urgency TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (status IN ('Pending', 'In Progress', 'Resolved')),
                customer_category TEXT NOT NULL DEFAULT '',
                record_type TEXT NOT NULL DEFAULT 'Customer',
                source_key TEXT UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_created_at
            ON tickets (created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_department
            ON tickets (department)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_urgency
            ON tickets (urgency)
            """
        )


def create_ticket(
    customer_name,
    email,
    ticket_text,
    department,
    urgency,
    customer_category="",
    status="Pending",
    record_type=CUSTOMER_RECORD,
    source_key=None,
    database_path=DEFAULT_DATABASE_PATH,
):
    if not customer_name or not str(customer_name).strip():
        raise ValueError("Customer name is required.")
    if not ticket_text or not str(ticket_text).strip():
        raise ValueError("Complaint description is required.")
    if status not in STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(STATUSES)}.")
    if record_type not in (CUSTOMER_RECORD, DEMO_RECORD):
        raise ValueError("Record type is not supported.")

    with _database_connection(database_path) as connection:
        if source_key:
            existing = connection.execute(
                "SELECT ticket_id FROM tickets WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            if existing:
                return existing[0]

        ticket_id = f"TKT-{uuid.uuid4().hex[:16].upper()}"
        connection.execute(
            """
            INSERT INTO tickets (
                ticket_id, customer_name, email, ticket_text, department,
                urgency, status, customer_category, record_type, source_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                str(customer_name).strip(),
                str(email or "").strip(),
                str(ticket_text).strip(),
                str(department).strip(),
                str(urgency).strip(),
                status,
                str(customer_category or "").strip(),
                record_type,
                source_key,
            ),
        )
        return ticket_id


def import_demo_tickets(records, database_path=DEFAULT_DATABASE_PATH):
    rows = []
    for record in records:
        source_key = record.get("source_key")
        if not source_key:
            raise ValueError("Each imported example requires a stable source key.")
        rows.append(
            (
                f"TKT-{uuid.uuid4().hex[:16].upper()}",
                "Dataset Import",
                "",
                str(record["ticket_text"]).strip(),
                str(record["department"]).strip(),
                str(record["urgency"]).strip(),
                "Pending",
                "",
                DEMO_RECORD,
                source_key,
            )
        )

    if not rows:
        return {"imported": 0, "skipped": 0}

    with _database_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO tickets (
                ticket_id, customer_name, email, ticket_text, department,
                urgency, status, customer_category, record_type, source_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        imported = connection.total_changes - before
    return {"imported": imported, "skipped": len(rows) - imported}


def get_all_tickets(database_path=DEFAULT_DATABASE_PATH):
    with _database_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT ticket_id, customer_name, email, ticket_text, department,
                   urgency, status, customer_category, record_type,
                   created_at, updated_at
            FROM tickets
            ORDER BY created_at DESC, ticket_id DESC
            """,
            connection,
        )


def get_ticket_by_id(ticket_id, database_path=DEFAULT_DATABASE_PATH):
    with _database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT ticket_id, customer_name, email, ticket_text, department,
                   urgency, status, customer_category, record_type,
                   created_at, updated_at
            FROM tickets
            WHERE ticket_id = ?
            """,
            (str(ticket_id),),
        ).fetchone()
    if row is None:
        return None
    columns = (
        "ticket_id",
        "customer_name",
        "email",
        "ticket_text",
        "department",
        "urgency",
        "status",
        "customer_category",
        "record_type",
        "created_at",
        "updated_at",
    )
    return dict(zip(columns, row))


def update_ticket_status(
    ticket_id, status, database_path=DEFAULT_DATABASE_PATH
):
    if status not in STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(STATUSES)}.")
    with _database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE tickets
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
            """,
            (status, str(ticket_id)),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Ticket {ticket_id} was not found.")


def delete_ticket(ticket_id, database_path=DEFAULT_DATABASE_PATH):
    with _database_connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM tickets WHERE ticket_id = ?",
            (str(ticket_id),),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Ticket {ticket_id} was not found.")


def get_dashboard_statistics(database_path=DEFAULT_DATABASE_PATH):
    with _database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE record_type = ?) AS total,
                COUNT(*) FILTER (WHERE record_type = ? AND status = 'Pending')
                    AS pending,
                COUNT(*) FILTER (WHERE record_type = ? AND status = 'In Progress')
                    AS in_progress,
                COUNT(*) FILTER (WHERE record_type = ? AND status = 'Resolved')
                    AS resolved,
                COUNT(*) FILTER (WHERE record_type = ? AND lower(urgency) = 'critical')
                    AS critical,
                COUNT(*) FILTER (WHERE record_type = ? AND lower(urgency) = 'high')
                    AS high,
                COUNT(*) FILTER (WHERE record_type = ?) AS demo_imported
            FROM tickets
            """,
            (
                CUSTOMER_RECORD,
                CUSTOMER_RECORD,
                CUSTOMER_RECORD,
                CUSTOMER_RECORD,
                CUSTOMER_RECORD,
                CUSTOMER_RECORD,
                DEMO_RECORD,
            ),
        ).fetchone()
    return dict(
        zip(
            (
                "total",
                "pending",
                "in_progress",
                "resolved",
                "critical",
                "high",
                "demo_imported",
            ),
            row,
        )
    )


def get_department_statistics(
    include_demo=False, database_path=DEFAULT_DATABASE_PATH
):
    return _get_group_statistics(
        "department",
        include_demo,
        database_path,
    )


def get_priority_statistics(
    include_demo=False, database_path=DEFAULT_DATABASE_PATH
):
    return _get_group_statistics(
        "urgency",
        include_demo,
        database_path,
    )


def _get_group_statistics(column, include_demo, database_path):
    if column not in ("department", "urgency"):
        raise ValueError("Unsupported statistics column.")
    with _database_connection(database_path) as connection:
        if include_demo:
            query = (
                f"SELECT {column} AS label, COUNT(*) AS ticket_count "
                f"FROM tickets GROUP BY {column} ORDER BY {column}"
            )
            rows = connection.execute(query).fetchall()
        else:
            query = (
                f"SELECT {column} AS label, COUNT(*) AS ticket_count "
                f"FROM tickets WHERE record_type = ? "
                f"GROUP BY {column} ORDER BY {column}"
            )
            rows = connection.execute(query, (CUSTOMER_RECORD,)).fetchall()
    return [{"label": row[0], "ticket_count": row[1]} for row in rows]
