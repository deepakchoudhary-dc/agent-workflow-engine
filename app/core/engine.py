"""
Workflow Engine.

The core graph-based workflow execution engine that supports:
- Nodes: Python functions that read/modify shared state
- Edges: Connections defining execution flow
- Branching: Conditional routing based on state
- Looping: Repeated execution until conditions are met
"""

from __future__ import annotations

import asyncio
import time
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

from .state import WorkflowState
from .tools import ToolRegistry, get_global_registry
from .storage import SQLiteStore
from . import sandbox as sandbox


class NodeType(str, Enum):
    """Types of nodes in a workflow."""

    STANDARD = "standard"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    START = "start"
    END = "end"


class Node:
    """
    A workflow node that executes a function and modifies state.

    Nodes can be:
    - Standard: Execute once and pass to next node
    - Conditional: Branch based on state values
    - Loop: Repeat until condition is met
    """

    def __init__(
        self,
        name: str,
        func: Optional[Callable[[WorkflowState], WorkflowState]] = None,
        node_type: NodeType = NodeType.STANDARD,
        condition: Optional[Callable[[WorkflowState], Union[bool, str]]] = None,
        max_iterations: int = 100,
    ):
        self.name = name
        self.func = func
        self.node_type = node_type
        self.condition = condition
        self.max_iterations = max_iterations

    async def execute(
        self, state: WorkflowState, tools: Optional[ToolRegistry] = None
    ) -> WorkflowState:
        """
        Execute the node function with the given state.

        Args:
            state: The current workflow state.
            tools: Optional tool registry for node use.

        Returns:
            The modified workflow state.
        """
        if self.func is None:
            return state

        input_snapshot = state.snapshot()
        state.current_node = self.name

        # Execute function (support both sync and async)
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(state)
        else:
            result = self.func(state)

        # Record step in history
        output_snapshot = state.snapshot()
        state.record_step(self.name, input_snapshot, output_snapshot)

        return result if result is not None else state

    def evaluate_condition(self, state: WorkflowState) -> Union[bool, str]:
        """
        Evaluate the node's condition function.

        For conditional nodes: Returns the name of the next node.
        For loop nodes: Returns True to continue looping, False to exit.
        """
        if self.condition is None:
            return True
        return self.condition(state)


class Edge:
    """
    An edge connecting two nodes in a workflow.

    Edges can be:
    - Simple: Always transition to target node
    - Conditional: Transition based on condition result
    """

    def __init__(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[WorkflowState], bool]] = None,
        label: Optional[str] = None,
    ):
        self.source = source
        self.target = target
        self.condition = condition
        self.label = label

    def should_traverse(self, state: WorkflowState) -> bool:
        """Check if this edge should be traversed."""
        if self.condition is None:
            return True
        return self.condition(state)


@dataclass
class BackgroundTaskHandle:
    """Unified handle for background workflow runs."""

    run_id: str
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


class WorkflowGraph:
    """
    A workflow graph definition containing nodes and edges.
    """

    def __init__(self, graph_id: Optional[str] = None, name: str = "workflow"):
        self.graph_id = graph_id or str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = {}  # source -> [edges]
        self.start_node: Optional[str] = None
        self.end_nodes: set[str] = set()

    def add_node(
        self,
        name: str,
        func: Optional[Callable] = None,
        node_type: NodeType = NodeType.STANDARD,
        condition: Optional[Callable] = None,
        max_iterations: int = 100,
    ) -> "WorkflowGraph":
        """Add a node to the graph."""
        node = Node(
            name=name,
            func=func,
            node_type=node_type,
            condition=condition,
            max_iterations=max_iterations,
        )
        self.nodes[name] = node

        if node_type == NodeType.START:
            self.start_node = name
        elif node_type == NodeType.END:
            self.end_nodes.add(name)

        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[WorkflowState], bool]] = None,
        label: Optional[str] = None,
    ) -> "WorkflowGraph":
        """Add an edge between two nodes."""
        edge = Edge(source=source, target=target, condition=condition, label=label)

        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(edge)

        return self

    def set_edges(self, edge_map: Dict[str, str]) -> "WorkflowGraph":
        """Set simple edges from a dictionary mapping."""
        for source, target in edge_map.items():
            self.add_edge(source, target)
        return self

    def get_next_node(self, current: str, state: WorkflowState) -> Optional[str]:
        """
        Determine the next node based on current node and state.

        Handles conditional routing by evaluating edge conditions.
        """
        current_node = self.nodes.get(current)
        if not current_node:
            return None

        # For conditional nodes, use the node's condition to determine next
        if current_node.node_type == NodeType.CONDITIONAL:
            result = current_node.evaluate_condition(state)
            if isinstance(result, str):
                return result

        # For loop nodes, check if we should continue or exit
        if current_node.node_type == NodeType.LOOP:
            should_continue = current_node.evaluate_condition(state)
            if should_continue and state.iteration < current_node.max_iterations:
                # Find the loop-back edge
                for edge in self.edges.get(current, []):
                    if edge.label == "loop":
                        return edge.target

        # Check regular edges
        edges = self.edges.get(current, [])
        for edge in edges:
            if edge.label != "loop" and edge.should_traverse(state):
                return edge.target

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [
                {
                    "name": n.name,
                    "type": n.node_type.value,
                    "max_iterations": n.max_iterations,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                }
                for edges in self.edges.values()
                for edge in edges
            ],
            "start_node": self.start_node,
            "end_nodes": list(self.end_nodes),
        }

    def validate(self) -> None:
        """Basic graph validation: start/end presence, edge integrity, reachability, simple cycle check."""
        if not self.nodes:
            raise ValueError("Graph has no nodes")
        if not self.start_node:
            raise ValueError("Graph missing start node")
        if not self.end_nodes:
            raise ValueError("Graph missing end node")
        if self.start_node not in self.nodes:
            raise ValueError("Start node not found in nodes")
        for end in self.end_nodes:
            if end not in self.nodes:
                raise ValueError(f"End node not found: {end}")
        # Edge integrity
        for source, edges in self.edges.items():
            if source not in self.nodes:
                raise ValueError(f"Edge source missing node: {source}")
            for edge in edges:
                if edge.target not in self.nodes:
                    raise ValueError(f"Edge target missing node: {edge.target}")
        # Reachability
        visited = set()
        stack = [self.start_node]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for edge in self.edges.get(node, []):
                stack.append(edge.target)
        unreachable = set(self.nodes.keys()) - visited
        if unreachable:
            raise ValueError(f"Unreachable nodes: {', '.join(sorted(unreachable))}")
        # Simple cycle detection excluding loop-labeled edges
        temp_mark = set()
        perm_mark = set()

        def visit(n: str) -> None:
            if n in perm_mark or n not in self.edges:
                return
            if n in temp_mark:
                raise ValueError("Cycle detected in graph")
            temp_mark.add(n)
            for edge in self.edges.get(n, []):
                if edge.label == "loop":
                    continue
                visit(edge.target)
            temp_mark.remove(n)
            perm_mark.add(n)

        visit(self.start_node)


class ExecutionLog:
    """Tracks workflow execution steps."""

    def __init__(self, run_id: str, graph_id: str):
        self.run_id = run_id
        self.graph_id = graph_id
        self.steps: List[Dict[str, Any]] = []
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None

    def log_step(
        self,
        node_name: str,
        iteration: int,
        status: str,
        message: Optional[str] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Log an execution step."""
        step = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": node_name,
            "iteration": iteration,
            "status": status,
            "message": message,
            "data": data,
        }
        self.steps.append(step)
        return step

    def complete(self) -> None:
        """Mark execution as complete."""
        self.completed_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary."""
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "steps": self.steps,
        }


class WorkflowEngine:
    """
    The main workflow execution engine.

    Executes workflow graphs by:
    1. Starting from the start node
    2. Executing each node's function
    3. Following edges based on conditions
    4. Handling loops and branches
    5. Stopping at end nodes or when no next node exists
    """

    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        store: Optional[SQLiteStore] = None,
        node_timeout: Optional[float] = 10.0,
        auto_resume_interrupted: bool = False,
    ):
        self.tools = tools or get_global_registry()
        self.store = store or SQLiteStore()
        self.node_timeout = node_timeout
        self._graphs: Dict[str, WorkflowGraph] = {}
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._background_tasks: Dict[str, "BackgroundTaskHandle"] = {}
        self._log_callbacks: List[Callable] = []
        self._lock = threading.Lock()

        # Dedicated background event loop for run_background submissions
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._start_background_loop, daemon=True
        )
        self._bg_thread.start()
        # Rehydrate persisted runs so APIs can report state after restarts.
        try:
            persisted = self.store.list_runs()
            for r in persisted:
                run_id = r.get("run_id")
                data = self.store.get_run(run_id)
                if not data:
                    continue
                # Recreate in-memory placeholders for runs
                state_payload = data.get("state") or {}
                state = WorkflowState(
                    run_id=run_id,
                    graph_id=data.get("graph_id"),
                    data=state_payload.get("data", {}),
                    started_at=None,
                    completed_at=None,
                )
                log = ExecutionLog(run_id, data.get("graph_id"))
                # If persisted status was 'running', mark as 'interrupted' to indicate it did not complete
                status = data.get("status")
                if status == "running":
                    status = "interrupted"
                    try:
                        self.store.update_run_status(run_id, status=status)
                    except Exception:
                        pass

                self._runs[run_id] = {"state": state, "log": log, "status": status}
        except Exception:
            # Don't block startup on persistence issues
            pass
        # Optionally resume interrupted runs
        self._auto_resume = bool(auto_resume_interrupted)
        if self._auto_resume:
            try:
                self._resume_interrupted_runs()
            except Exception:
                pass

    def _resume_interrupted_runs(self) -> None:
        """Resume runs that were marked interrupted on startup by re-submitting them.

        This schedules background executions for runs with status == 'interrupted'.
        """

        # Run resume attempts in a separate thread so startup is not blocked.
        def _worker() -> None:
            with self._lock:
                to_resume = [
                    rid
                    for rid, data in self._runs.items()
                    if data.get("status") == "interrupted"
                ]

            for run_id in to_resume:
                entry = self._runs.get(run_id)
                if not entry:
                    continue
                state = entry.get("state")
                if not state:
                    continue

                # Do not duplicate resume if already scheduled
                with self._lock:
                    if run_id in self._background_tasks:
                        continue

                max_attempts = 3
                attempt = 0
                while attempt < max_attempts:
                    try:
                        # mark as resuming in-memory and persist
                        with self._lock:
                            self._runs[run_id]["status"] = "resuming"
                        try:
                            self.store.update_run_status(run_id, status="resuming")
                        except Exception:
                            pass

                        # Submit run in background using the original run_id
                        self.run_background(
                            graph_id=state.graph_id,
                            initial_state=state.data,
                            run_id=run_id,
                        )
                        break
                    except Exception:
                        attempt += 1
                        backoff = 2**attempt
                        time.sleep(backoff)

                # If exceeded attempts without scheduling, mark as resume_failed
                with self._lock:
                    if (
                        run_id not in self._background_tasks
                        and self._runs.get(run_id, {}).get("status") == "resuming"
                    ):
                        self._runs[run_id]["status"] = "resume_failed"
                        try:
                            self.store.update_run_status(run_id, status="resume_failed")
                        except Exception:
                            pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def register_graph(self, graph: WorkflowGraph) -> str:
        """Register a workflow graph (in memory and persisted)."""
        graph.validate()
        with self._lock:
            self._graphs[graph.graph_id] = graph
        # Persist lightweight definition (structure only)
        try:
            self.store.save_graph(graph.graph_id, graph.name, graph.to_dict())
        except Exception:
            # Persistence failures should not break execution; log silently
            pass
        return graph.graph_id

    def get_graph(self, graph_id: str) -> Optional[WorkflowGraph]:
        """Get a registered graph by ID."""
        with self._lock:
            return self._graphs.get(graph_id)

    def on_log(self, callback: Callable) -> None:
        """Register a callback for log events."""
        self._log_callbacks.append(callback)

    def _start_background_loop(self) -> None:
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    async def _emit_log(self, step: Dict[str, Any]) -> None:
        """Emit a log event to all registered callbacks."""
        for callback in self._log_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(step)
            else:
                callback(step)

    async def _execute_node(self, node: Node, state: WorkflowState) -> WorkflowState:
        """Execute a node with optional timeout, supporting sync and async functions."""
        timeout = self.node_timeout

        if node.func is None:
            return state

        async def run_callable() -> WorkflowState:
            if asyncio.iscoroutinefunction(node.func):
                return await node.func(state)
            return await asyncio.to_thread(node.func, state)

        if timeout:
            return await asyncio.wait_for(run_callable(), timeout=timeout)
        return await run_callable()

    async def run(
        self,
        graph_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow graph.

        Args:
            graph_id: The ID of the graph to execute.
            initial_state: Initial state data.
            run_id: Optional run ID. Generated if not provided.

        Returns:
            Dictionary containing final state and execution log.
        """
        graph = self._graphs.get(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")

        run_id = run_id or str(uuid.uuid4())
        state = WorkflowState(
            run_id=run_id,
            graph_id=graph_id,
            data=initial_state or {},
            started_at=datetime.now(timezone.utc),
        )

        log = ExecutionLog(run_id, graph_id)

        # Store run state
        with self._lock:
            self._runs[run_id] = {
                "state": state,
                "log": log,
                "status": "running",
            }
        try:
            self.store.save_run(
                run_id=run_id,
                graph_id=graph_id,
                status="running",
                state=state.to_output(),
                log=log.to_dict(),
                started_at=state.started_at.isoformat() if state.started_at else None,
                completed_at=None,
            )
        except Exception:
            pass

        try:
            # Start execution from start node
            current_node_name = graph.start_node
            if not current_node_name:
                # Find first node if no explicit start
                current_node_name = next(iter(graph.nodes.keys()), None)

            if not current_node_name:
                raise ValueError("No nodes in graph")

            step = log.log_step(
                current_node_name, state.iteration, "started", "Workflow started"
            )
            await self._emit_log(step)

            visited_states: Dict[str, int] = {}  # Track visits for loop detection

            while current_node_name:
                node = graph.nodes.get(current_node_name)
                if not node:
                    break

                # Track iterations for current node
                visit_key = f"{current_node_name}"
                visited_states[visit_key] = visited_states.get(visit_key, 0) + 1

                # Check max iterations for safety
                if visited_states[visit_key] > node.max_iterations:
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "max_iterations",
                        f"Max iterations ({node.max_iterations}) reached for node",
                    )
                    await self._emit_log(step)
                    break

                # Execute the node
                step = log.log_step(
                    current_node_name,
                    state.iteration,
                    "executing",
                    f"Executing node: {current_node_name}",
                )
                await self._emit_log(step)

                try:
                    state = await self._execute_node(node, state)
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "completed",
                        f"Node completed: {current_node_name}",
                        data=state.snapshot(),
                    )
                    await self._emit_log(step)
                except asyncio.TimeoutError:
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "timeout",
                        f"Node timed out: {current_node_name}",
                    )
                    await self._emit_log(step)
                    state.error = f"Node timed out: {current_node_name}"
                    break
                except Exception as e:
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "error",
                        f"Error in node {current_node_name}: {str(e)}",
                    )
                    await self._emit_log(step)
                    state.error = str(e)
                    break

                # Check if we're at an end node
                if current_node_name in graph.end_nodes:
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "end_reached",
                        "Reached end node",
                    )
                    await self._emit_log(step)
                    break

                # Determine next node
                next_node_name = graph.get_next_node(current_node_name, state)

                if next_node_name:
                    step = log.log_step(
                        current_node_name,
                        state.iteration,
                        "transition",
                        f"Transitioning from {current_node_name} to {next_node_name}",
                    )
                    await self._emit_log(step)

                current_node_name = next_node_name
                state.iteration += 1

            # Mark completion
            state.completed_at = datetime.now(timezone.utc)
            log.complete()

            step = log.log_step(
                "workflow", state.iteration, "completed", "Workflow completed"
            )
            await self._emit_log(step)

            with self._lock:
                self._runs[run_id]["status"] = "completed"
            try:
                self.store.update_run_status(
                    run_id,
                    status="completed",
                    state=state.to_output(),
                    log=log.to_dict(),
                    completed_at=(
                        state.completed_at.isoformat() if state.completed_at else None
                    ),
                )
            except Exception:
                pass

        except Exception as e:
            state.error = str(e)
            state.completed_at = datetime.now(timezone.utc)
            log.log_step(
                "workflow", state.iteration, "error", f"Workflow error: {str(e)}"
            )
            log.complete()
            with self._lock:
                self._runs[run_id]["status"] = "error"
            try:
                self.store.update_run_status(
                    run_id,
                    status="error",
                    state=state.to_output(),
                    log=log.to_dict(),
                    completed_at=(
                        state.completed_at.isoformat() if state.completed_at else None
                    ),
                )
            except Exception:
                pass

        with self._lock:
            status = self._runs[run_id]["status"]
        return {
            "run_id": run_id,
            "graph_id": graph_id,
            "status": status,
            "final_state": state.to_output(),
            "execution_log": log.to_dict(),
        }

    def run_background(
        self,
        graph_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Start a workflow run in the background and return run_id immediately."""
        run_id = run_id or str(uuid.uuid4())

        # Persist a lightweight run record immediately so the run can be
        # queried or cancelled even if the in-memory background handle is
        # not present (e.g. across process restarts).
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.store.save_run(
                run_id=run_id,
                graph_id=graph_id,
                status="running",
                state={
                    "run_id": run_id,
                    "graph_id": graph_id,
                    "data": initial_state or {},
                },
                log={"run_id": run_id, "graph_id": graph_id, "steps": []},
                started_at=now_iso,
                completed_at=None,
            )
        except Exception:
            pass

        async def _task():
            return await self.run(
                graph_id=graph_id, initial_state=initial_state, run_id=run_id
            )

        try:
            loop = asyncio.get_running_loop()
            future = loop.create_task(_task())
            handle = BackgroundTaskHandle(run_id=run_id, future=future, loop=loop)
        except RuntimeError:
            future = asyncio.run_coroutine_threadsafe(_task(), self._bg_loop)
            handle = BackgroundTaskHandle(
                run_id=run_id, future=future, loop=self._bg_loop
            )

        with self._lock:
            self._background_tasks[run_id] = handle

        # Also ensure an in-memory run placeholder exists so immediate state queries
        # return something useful while the background coroutine updates the run.
        with self._lock:
            if run_id not in self._runs:
                from .state import WorkflowState

                placeholder = WorkflowState(
                    run_id=run_id, graph_id=graph_id, data=initial_state or {}
                )
                self._runs[run_id] = {
                    "state": placeholder,
                    "log": ExecutionLog(run_id, graph_id),
                    "status": "running",
                }

        return run_id

    def cancel_run(self, run_id: str) -> bool:
        """Attempt to cancel a background run."""
        with self._lock:
            handle = self._background_tasks.get(run_id)
        if not handle:
            # If there's no in-memory handle, try to mark a persisted run as cancelled
            # Also attempt to signal any registered subprocesses for this run.
            try:
                sandbox.cancel_run(run_id)
            except Exception:
                pass

            persisted = self.store.get_run(run_id)
            if persisted and persisted.get("status") == "running":
                try:
                    self.store.update_run_status(run_id, status="cancelled")
                except Exception:
                    pass
                with self._lock:
                    if run_id in self._runs:
                        self._runs[run_id]["status"] = "cancelled"
                return True
            return False
        # Signal subprocesses (if any) before cancelling the coroutine/future.
        try:
            sandbox.cancel_run(run_id)
        except Exception:
            pass

        cancelled = handle.future.cancel()
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["status"] = "cancelled"
            self.store.update_run_status(run_id, status="cancelled")
        return cancelled

    def get_background_status(self, run_id: str) -> Optional[str]:
        with self._lock:
            handle = self._background_tasks.get(run_id)
        if not handle:
            # Fall back to persisted run status
            persisted = self.store.get_run(run_id)
            if not persisted:
                return None
            return persisted.get("status")
        fut = handle.future
        if fut.cancelled():
            return "cancelled"
        if fut.done():
            return "completed"
        return "running"

    def get_run_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a run."""
        with self._lock:
            run = self._runs.get(run_id)
        if run:
            state: WorkflowState = run["state"]
            log: ExecutionLog = run["log"]
            return {
                "run_id": run_id,
                "status": run["status"],
                "current_state": state.to_output(),
                "execution_log": log.to_dict(),
            }

        # Fallback to persisted run
        persisted = self.store.get_run(run_id)
        if not persisted:
            return None
        return {
            "run_id": run_id,
            "status": persisted["status"],
            "current_state": persisted.get("state"),
            "execution_log": persisted.get("log"),
        }

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all registered graphs."""
        return [graph.to_dict() for graph in self._graphs.values()]

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with their status."""
        with self._lock:
            in_memory = [
                {
                    "run_id": run_id,
                    "graph_id": data["state"].graph_id,
                    "status": data["status"],
                }
                for run_id, data in self._runs.items()
            ]
        persisted = self.store.list_runs()
        # Merge unique run_ids (in-memory overrides persisted duplicates)
        seen = {r["run_id"] for r in in_memory}
        merged = in_memory + [r for r in persisted if r["run_id"] not in seen]
        return merged
