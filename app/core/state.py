"""
Workflow State Management.

Provides a typed state container that flows through workflow nodes,
supporting both dictionary-style and attribute-style access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from copy import deepcopy


class WorkflowState(BaseModel):
    """
    A Pydantic model representing the shared state that flows through workflow nodes.

    Supports both structured fields and a flexible data dictionary for arbitrary values.
    All state changes are tracked with timestamps.
    """

    # Core workflow metadata
    run_id: Optional[str] = None
    graph_id: Optional[str] = None
    current_node: Optional[str] = None
    iteration: int = 0

    # Flexible data store for workflow-specific values
    data: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    history: list[Dict[str, Any]] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    model_config = {"extra": "allow"}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dictionary."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the data dictionary."""
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple values in the data dictionary."""
        self.data.update(values)

    def record_step(
        self, node_name: str, input_snapshot: Dict, output_snapshot: Dict
    ) -> None:
        """Record a step in the execution history."""
        self.history.append(
            {
                "node": node_name,
                "iteration": self.iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": input_snapshot,
                "output": output_snapshot,
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current data state."""
        return deepcopy(self.data)

    def to_output(self) -> Dict[str, Any]:
        """Convert state to a clean output format."""
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "final_node": self.current_node,
            "iterations": self.iteration,
            "data": self.data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error": self.error,
            "execution_log": self.history,
        }


class StateManager:
    """
    Manages workflow states across multiple runs.

    Provides thread-safe storage and retrieval of workflow states.
    """

    def __init__(self):
        self._states: Dict[str, WorkflowState] = {}

    def create(
        self, run_id: str, graph_id: str, initial_data: Dict[str, Any]
    ) -> WorkflowState:
        """Create a new workflow state."""
        state = WorkflowState(
            run_id=run_id,
            graph_id=graph_id,
            data=initial_data,
            started_at=datetime.now(timezone.utc),
        )
        self._states[run_id] = state
        return state

    def get(self, run_id: str) -> Optional[WorkflowState]:
        """Retrieve a workflow state by run ID."""
        return self._states.get(run_id)

    def update(self, run_id: str, state: WorkflowState) -> None:
        """Update a workflow state."""
        self._states[run_id] = state

    def complete(self, run_id: str) -> None:
        """Mark a workflow state as completed."""
        if run_id in self._states:
            self._states[run_id].completed_at = datetime.now(timezone.utc)

    def set_error(self, run_id: str, error: str) -> None:
        """Set an error on a workflow state."""
        if run_id in self._states:
            self._states[run_id].error = error
            self._states[run_id].completed_at = datetime.now(timezone.utc)

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        return list(self._states.keys())
