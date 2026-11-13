"""The mock CRM's storage: one sqlite row per lead.

A lead is keyed by its id, so saving the same lead again updates its record and
never adds a second one. The file path lets state survive between MCP server
processes. The default is in memory, for tests.
"""

import sqlite3
import threading
from pathlib import Path


class CrmStore:
    def __init__(self, path: str | Path = ":memory:"):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS leads (lead_id TEXT PRIMARY KEY, label TEXT, notes TEXT, outreach TEXT)"
            )

    def upsert_lead(self, lead_id: str, label: str, notes: str) -> dict:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO leads (lead_id, label, notes) VALUES (?, ?, ?) "
                "ON CONFLICT(lead_id) DO UPDATE SET label = excluded.label, notes = excluded.notes",
                (lead_id, label, notes),
            )
        return self.get_lead(lead_id)

    def record_outreach(self, lead_id: str, draft_text: str) -> dict:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO leads (lead_id, outreach) VALUES (?, ?) "
                "ON CONFLICT(lead_id) DO UPDATE SET outreach = excluded.outreach",
                (lead_id, draft_text),
            )
        return self.get_lead(lead_id)

    def get_lead(self, lead_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()
        return dict(row) if row else None

    def list_records(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM leads ORDER BY lead_id")]
