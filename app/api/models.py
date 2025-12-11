"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Graph Creation Models
# ============================================================================


class NodeDefinition(BaseModel):
    """Definition of a node in a workflow graph."""

    name: str = Field(..., description="Unique name for the node")
    type: str = Field(
        default="standard",
        description="Node type: standard, conditional, loop, start, end",
    )
    function: Optional[str] = Field(
        default=None, description="Name of registered function to execute"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Name of condition function for conditional/loop nodes",
    )
    max_iterations: int = Field(
        default=100, description="Maximum iterations for loop nodes"
    )


class EdgeDefinition(BaseModel):
    """Definition of an edge connecting two nodes."""

    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    condition: Optional[str] = Field(
        default=None, description="Optional condition function name"
    )
    label: Optional[str] = Field(
        default=None, description="Optional label (e.g., 'loop' for loop-back edges)"
    )


class CreateGraphRequest(BaseModel):
    """Request body for creating a new workflow graph."""

    name: str = Field(default="workflow", description="Name of the workflow")
    nodes: List[NodeDefinition] = Field(..., description="List of node definitions")
    edges: List[EdgeDefinition] = Field(..., description="List of edge definitions")
    start_node: Optional[str] = Field(
        default=None,
        description="Name of the start node (optional, first node used if not specified)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "simple_workflow",
                    "nodes": [
                        {"name": "start", "type": "start"},
                        {
                            "name": "process",
                            "type": "standard",
                            "function": "process_data",
                        },
                        {"name": "end", "type": "end"},
                    ],
                    "edges": [
                        {"source": "start", "target": "process"},
                        {"source": "process", "target": "end"},
                    ],
                }
            ]
        }
    }


class CreateGraphResponse(BaseModel):
    """Response for graph creation."""

    graph_id: str = Field(..., description="Unique identifier for the created graph")
    name: str = Field(..., description="Name of the workflow")
    node_count: int = Field(..., description="Number of nodes in the graph")
    edge_count: int = Field(..., description="Number of edges in the graph")


# ============================================================================
# Graph Execution Models
# ============================================================================


class RunGraphRequest(BaseModel):
    """Request body for running a workflow graph."""

    graph_id: str = Field(..., description="ID of the graph to run")
    initial_state: Dict[str, Any] = Field(
        default_factory=dict, description="Initial state data for the workflow"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "graph_id": "abc123",
                    "initial_state": {
                        "code": "def hello(): pass",
                        "quality_threshold": 7,
                    },
                }
            ]
        }
    }


class ExecutionStep(BaseModel):
    """A single step in the execution log."""

    timestamp: str
    node: str
    iteration: int
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ExecutionLog(BaseModel):
    """Execution log for a workflow run."""

    run_id: str
    graph_id: str
    started_at: str
    completed_at: Optional[str] = None
    steps: List[ExecutionStep]


class RunGraphResponse(BaseModel):
    """Response for graph execution."""

    run_id: str = Field(..., description="Unique identifier for this run")
    graph_id: str = Field(..., description="ID of the executed graph")
    status: str = Field(..., description="Execution status: running, completed, error")
    final_state: Dict[str, Any] = Field(..., description="Final workflow state")
    execution_log: ExecutionLog = Field(..., description="Detailed execution log")


# ============================================================================
# State Query Models
# ============================================================================


class GetStateResponse(BaseModel):
    """Response for state queries."""

    run_id: str
    status: str
    current_state: Dict[str, Any]
    execution_log: ExecutionLog


# ============================================================================
# Tool Models
# ============================================================================


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    annotation: str
    default: Optional[Any] = None
    required: bool


class ToolInfo(BaseModel):
    """Information about a registered tool."""

    name: str
    description: str
    parameters: Dict[str, ToolParameter]


class RegisterToolRequest(BaseModel):
    """Request to register a new tool (for dynamic registration)."""

    name: str = Field(..., description="Unique name for the tool")
    code: str = Field(..., description="Python code defining the tool function")
    description: Optional[str] = Field(default=None, description="Tool description")


class ListToolsResponse(BaseModel):
    """Response listing all registered tools."""

    tools: List[ToolInfo]


# ============================================================================
# Graph Info Models
# ============================================================================


class NodeInfo(BaseModel):
    """Information about a node in a graph."""

    name: str
    type: str
    max_iterations: int


class EdgeInfo(BaseModel):
    """Information about an edge in a graph."""

    source: str
    target: str
    label: Optional[str] = None


class GraphInfo(BaseModel):
    """Detailed information about a graph."""

    graph_id: str
    name: str
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]
    start_node: Optional[str]
    end_nodes: List[str]


class ListGraphsResponse(BaseModel):
    """Response listing all registered graphs."""

    graphs: List[GraphInfo]


class RunInfo(BaseModel):
    """Brief information about a workflow run."""

    run_id: str
    graph_id: str
    status: str


class ListRunsResponse(BaseModel):
    """Response listing all runs."""

    runs: List[RunInfo]


# ============================================================================
# Error Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        default=None, description="Detailed error information"
    )
