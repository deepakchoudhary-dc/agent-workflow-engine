"""
FastAPI Routes for the Workflow Engine.

Provides REST API endpoints for:
- Creating workflow graphs
- Running workflows
- Querying workflow state
- Managing tools
"""

import ast
import inspect
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .models import (
    CreateGraphRequest,
    CreateGraphResponse,
    RunGraphRequest,
    RunGraphResponse,
    GetStateResponse,
    ListGraphsResponse,
    ListToolsResponse,
    ListRunsResponse,
    GraphInfo,
    RegisterToolRequest,
    ErrorResponse,
)
from ..core.engine import WorkflowEngine, WorkflowGraph, NodeType
import os
from ..core.tools import get_global_registry
from ..core.sandbox import RemoteFunction


# Create router
router = APIRouter(prefix="/graph", tags=["Workflow Graph"])

# Global engine instance (initialized in main.py)
_engine: WorkflowEngine = None
_node_functions: Dict[str, callable] = {}


def _safe_exec(code: str, local_ns: Dict[str, Any]) -> None:
    """Execute user code in a restricted namespace with basic AST checks.

    This is a lightweight safety layer — it prevents imports and some
    obviously dangerous constructs, and limits available builtins. It is
    NOT a full sandbox; for untrusted code run in a separate process.
    """
    # Basic AST validation: disallow Import, ImportFrom, Exec, and Name('__import__')
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        raise

    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in dynamic code")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
        ):
            raise ValueError("Use of __import__ is not allowed")

    # Restrict builtins to safe subset
    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "sorted": sorted,
        "enumerate": enumerate,
        "map": map,
        "filter": filter,
        "zip": zip,
        "abs": abs,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
    }

    restricted_globals = {"__builtins__": safe_builtins, "time": time}
    # Small guard: avoid huge payloads
    if len(code) > 20000:
        raise ValueError("Code payload too large")

    exec(
        compile(parsed, filename="<user_code>", mode="exec"),
        restricted_globals,
        local_ns,
    )


def get_engine() -> WorkflowEngine:
    """Get the workflow engine instance."""
    global _engine
    if _engine is None:
        auto = os.environ.get("AUTO_RESUME", "0") in ("1", "true", "True")
        _engine = WorkflowEngine(auto_resume_interrupted=auto)
    return _engine


def register_node_function(name: str, func: callable) -> None:
    """Register a function that can be used by nodes."""
    _node_functions[name] = func


def get_node_function(name: str) -> callable:
    """Get a registered node function by name."""
    return _node_functions.get(name)


def resolve_callable(name: str) -> Optional[callable]:
    """Resolve a callable from registered node functions or tools."""
    func = get_node_function(name)
    if func:
        return func
    registry = get_global_registry()
    return registry.get(name)


# ============================================================================
# Graph Management Endpoints
# ============================================================================


@router.post(
    "/create",
    response_model=CreateGraphResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Create a new workflow graph",
    description="Create a new workflow graph from node and edge definitions.",
)
async def create_graph(request: CreateGraphRequest) -> CreateGraphResponse:
    """
    Create a new workflow graph.

    The graph is defined by:
    - **nodes**: List of node definitions with names, types, and optional functions
    - **edges**: List of edges connecting nodes
    - **start_node**: Optional starting node (defaults to first node)
    """
    engine = get_engine()

    try:
        # Create the graph
        graph = WorkflowGraph(name=request.name)

        # Add nodes
        for node_def in request.nodes:
            node_type = NodeType(node_def.type) if node_def.type else NodeType.STANDARD

            # Get function if specified
            func = None
            if node_def.function:
                func = resolve_callable(node_def.function)
                if func is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Function not found: {node_def.function}",
                    )

            # Get condition if specified
            condition = None
            if node_def.condition:
                condition = resolve_callable(node_def.condition)

            graph.add_node(
                name=node_def.name,
                func=func,
                node_type=node_type,
                condition=condition,
                max_iterations=node_def.max_iterations,
            )

        # Set start node
        if request.start_node:
            if request.start_node not in graph.nodes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Start node not found: {request.start_node}",
                )
            graph.start_node = request.start_node
        elif request.nodes:
            graph.start_node = request.nodes[0].name

        # Add edges
        for edge_def in request.edges:
            if edge_def.source not in graph.nodes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Edge source node not found: {edge_def.source}",
                )
            if edge_def.target not in graph.nodes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Edge target node not found: {edge_def.target}",
                )

            condition = None
            if edge_def.condition:
                condition = resolve_callable(edge_def.condition)

            graph.add_edge(
                source=edge_def.source,
                target=edge_def.target,
                condition=condition,
                label=edge_def.label,
            )

        # Register the graph
        engine.register_graph(graph)

        return CreateGraphResponse(
            graph_id=graph.graph_id,
            name=graph.name,
            node_count=len(graph.nodes),
            edge_count=sum(len(edges) for edges in graph.edges.values()),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/run",
    response_model=RunGraphResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Run a workflow graph",
    description="Execute a workflow graph with the given initial state.",
)
async def run_graph(request: RunGraphRequest) -> RunGraphResponse:
    """
    Run a workflow graph.

    Executes the graph identified by **graph_id** with the provided **initial_state**.
    Returns the final state and a detailed execution log.
    """
    engine = get_engine()

    # Check if graph exists
    graph = engine.get_graph(request.graph_id)
    if not graph:
        raise HTTPException(
            status_code=404, detail=f"Graph not found: {request.graph_id}"
        )

    try:
        # Run the workflow
        result = await engine.run(
            graph_id=request.graph_id,
            initial_state=request.initial_state,
        )

        return RunGraphResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/run_async",
    response_model=Dict[str, str],
    responses={404: {"model": ErrorResponse}},
    summary="Run a workflow graph in background",
    description=(
        "Start execution and return run_id immediately. "
        "Poll /graph/state/{run_id} or use websocket for logs."
    ),
)
async def run_graph_async(request: RunGraphRequest) -> Dict[str, str]:
    engine = get_engine()

    graph = engine.get_graph(request.graph_id)
    if not graph:
        raise HTTPException(
            status_code=404, detail=f"Graph not found: {request.graph_id}"
        )

    run_id = engine.run_background(
        graph_id=request.graph_id, initial_state=request.initial_state
    )
    return {"run_id": run_id, "status": "running"}


@router.get(
    "/state/{run_id}",
    response_model=GetStateResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get workflow run state",
    description="Get the current state of an ongoing or completed workflow run.",
)
async def get_state(run_id: str) -> GetStateResponse:
    """
    Get the current state of a workflow run.

    Returns the current state and execution log for the specified **run_id**.
    """
    engine = get_engine()

    state = engine.get_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return GetStateResponse(**state)


@router.get(
    "/list",
    response_model=ListGraphsResponse,
    summary="List all graphs",
    description="Get a list of all registered workflow graphs.",
)
async def list_graphs() -> ListGraphsResponse:
    """List all registered workflow graphs."""
    engine = get_engine()
    graphs = engine.list_graphs()

    return ListGraphsResponse(graphs=[GraphInfo(**g) for g in graphs])


@router.get(
    "/runs",
    response_model=ListRunsResponse,
    summary="List all runs",
    description="Get a list of all workflow runs.",
)
async def list_runs() -> ListRunsResponse:
    """List all workflow runs."""
    engine = get_engine()
    return ListRunsResponse(runs=engine.list_runs())


@router.get(
    "/status/{run_id}",
    response_model=Dict[str, str],
    responses={404: {"model": ErrorResponse}},
    summary="Get run status",
    description="Get background run status (running/completed/cancelled).",
)
async def get_status(run_id: str) -> Dict[str, str]:
    engine = get_engine()
    status = engine.get_background_status(run_id)
    if status is None:
        state = engine.get_run_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        status = state.get("status", "unknown")
    return {"run_id": run_id, "status": status}


@router.post(
    "/cancel/{run_id}",
    response_model=Dict[str, str],
    responses={404: {"model": ErrorResponse}},
    summary="Cancel a background run",
    description="Attempt to cancel a background workflow run.",
)
async def cancel_run(run_id: str) -> Dict[str, str]:
    engine = get_engine()
    success = engine.cancel_run(run_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Run not found or not cancellable: {run_id}"
        )
    return {"run_id": run_id, "status": "cancelled"}


@router.get(
    "/{graph_id}",
    response_model=GraphInfo,
    responses={404: {"model": ErrorResponse}},
    summary="Get graph details",
    description="Get detailed information about a specific graph.",
)
async def get_graph(graph_id: str) -> GraphInfo:
    """Get details of a specific graph."""
    engine = get_engine()

    graph = engine.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")

    return GraphInfo(**graph.to_dict())


# ============================================================================
# Tool Management Endpoints
# ============================================================================

tool_router = APIRouter(prefix="/tools", tags=["Tools"])
node_router = APIRouter(prefix="/nodes", tags=["Nodes"])


@tool_router.get(
    "/list",
    response_model=ListToolsResponse,
    summary="List all tools",
    description="Get a list of all registered tools.",
)
async def list_tools() -> ListToolsResponse:
    """List all registered tools."""
    registry = get_global_registry()
    return ListToolsResponse(tools=registry.list_tools())


@tool_router.post(
    "/register",
    response_model=Dict[str, str],
    responses={400: {"model": ErrorResponse}},
    summary="Register a new tool",
    description="Dynamically register a Python function as a tool.",
)
async def register_tool(request: RegisterToolRequest) -> Dict[str, str]:
    """Register a tool from provided Python code string."""
    registry = get_global_registry()
    local_ns: Dict[str, Any] = {}
    try:
        _safe_exec(request.code, local_ns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to execute code: {exc}")

    func = local_ns.get(request.name)
    if not func or not inspect.isfunction(func):
        raise HTTPException(
            status_code=400, detail="No function found matching the provided name"
        )

    registry.add(
        func,
        name=request.name,
        description=(request.description or (func.__doc__ or "")),
    )
    return {"status": "registered", "name": request.name}


@node_router.post(
    "/register",
    response_model=Dict[str, str],
    responses={400: {"model": ErrorResponse}},
    summary="Register a node function",
    description="Register a Python function to be used as a workflow node or condition.",
)
async def register_node(request: RegisterToolRequest) -> Dict[str, str]:
    local_ns: Dict[str, Any] = {}
    try:
        _safe_exec(request.code, local_ns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to execute code: {exc}")
    # If code executed successfully, prefer wrapping as a subprocess-backed RemoteFunction
    # to reduce risk of running untrusted code in-process.
    if request.name not in local_ns or not inspect.isfunction(
        local_ns.get(request.name)
    ):
        raise HTTPException(
            status_code=400, detail="No function found matching the provided name"
        )

    # Keep original code string and register a RemoteFunction wrapper
    remote = RemoteFunction(code=request.code, name=request.name, timeout=5.0)
    register_node_function(request.name, remote)
    return {"status": "registered", "name": request.name}


# ============================================================================
# WebSocket Endpoint for Real-time Logs
# ============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time log streaming."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/{run_id}")
async def websocket_logs(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for streaming execution logs.

    Connect to receive real-time updates for a specific workflow run.
    """
    await manager.connect(websocket, run_id)

    try:
        while True:
            # Keep connection alive, wait for messages
            data = await websocket.receive_text()

            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)


def get_websocket_manager() -> ConnectionManager:
    """Get the WebSocket connection manager."""
    return manager
