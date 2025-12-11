"""Simple SQLite persistence for graphs and runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "workflow.db"


class SQLiteStore:
    """Lightweight SQLite-backed storage for graphs and runs."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._ensure_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graphs (
                    graph_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT,
                    log_json TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(graph_id) REFERENCES graphs(graph_id)
                )
                """
            )
            conn.commit()

    def save_graph(self, graph_id: str, name: str, definition: Dict[str, Any]) -> None:
        payload = json.dumps(definition)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO graphs (graph_id, name, definition, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (graph_id, name, payload, now),
            )
            conn.commit()

    def load_graphs(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT graph_id, name, definition FROM graphs"
            ).fetchall()
        graphs: List[Dict[str, Any]] = []
        for graph_id, name, definition in rows:
            try:
                graphs.append(json.loads(definition))
            except json.JSONDecodeError:
                continue
        return graphs

    def save_run(
        self,
        run_id: str,
        graph_id: str,
        status: str,
        state: Dict[str, Any],
        log: Dict[str, Any],
        started_at: Optional[str],
        completed_at: Optional[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, graph_id, status, state_json, log_json, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    graph_id,
                    status,
                    json.dumps(state),
                    json.dumps(log),
                    started_at,
                    completed_at,
                ),
            )
            conn.commit()

    def update_run_status(
        self,
        run_id: str,
        status: str,
        state: Optional[Dict[str, Any]] = None,
        log: Optional[Dict[str, Any]] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    state_json = COALESCE(?, state_json),
                    log_json = COALESCE(?, log_json),
                    completed_at = COALESCE(?, completed_at)
                WHERE run_id = ?
                """,
                (
                    status,
                    json.dumps(state) if state is not None else None,
                    json.dumps(log) if log is not None else None,
                    completed_at,
                    run_id,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT run_id, graph_id, status, state_json, log_json, started_at, completed_at
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        run_id, graph_id, status, state_json, log_json, started_at, completed_at = row
        try:
            state = json.loads(state_json) if state_json else None
            log = json.loads(log_json) if log_json else None
        except json.JSONDecodeError:
            state, log = None, None
        return {
            "run_id": run_id,
            "graph_id": graph_id,
            "status": status,
            "state": state,
            "log": log,
            "started_at": started_at,
            "completed_at": completed_at,
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, graph_id, status, started_at, completed_at
                FROM runs
                ORDER BY started_at DESC
                """
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "graph_id": r[1],
                "status": r[2],
                "started_at": r[3],
                "completed_at": r[4],
            }
            for r in rows
        ]
