# Agent Workflow Engine

A minimal graph-based workflow engine with a FastAPI HTTP API.

Features
- Graphs: nodes (Python functions), edges, branching and loops
- Shared state: `WorkflowState` flows through nodes
- Tool registry: register reusable functions via API
- Background runs: async background execution with run status and cancellation
- WebSocket: simple log streaming per run

Quickstart

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
uvicorn app.main:app --reload
```

3. Use the API (examples):
- `POST /nodes/register` to register node functions
- `POST /tools/register` to register helper tools
- `POST /graph/create` to create a workflow
- `POST /graph/run` to run synchronously
- `POST /graph/run_async` to run in background
- `GET /graph/state/{run_id}` to fetch run state
- `POST /graph/cancel/{run_id}` to attempt cancellation
Agent Workflow Engine

A minimal graph-based workflow engine with a FastAPI HTTP API.

Features
- Graphs: nodes (Python functions), edges, basic branching and loops
- Shared state: `WorkflowState` flows through nodes (dict-like API)
- Tool registry: register reusable functions via API
- Background runs: async background execution with run status and cancellation
- WebSocket: simple log streaming per run

Quickstart

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
uvicorn app.main:app --reload
```

API Examples (curl)

- Register a node function (node will be executed in a subprocess for isolation):

```bash
curl -X POST http://localhost:8000/nodes/register \
  -H "Content-Type: application/json" \
  -d '{"name":"increment","code":"def increment(state):\n  value = state.get(\"value\", 0)\n  state.set(\"value\", value+1)\n  return state"}'
```

- Register a tool (reusable function available to nodes):

```bash
curl -X POST http://localhost:8000/tools/register \
  -H "Content-Type: application/json" \
  -d '{"name":"add_one","code":"def add_one(state):\n  v = state.get(\"n\",0)\n  state.set(\"n\", v+1)\n  return state"}'
```

- Create a simple linear graph:

```bash
curl -X POST http://localhost:8000/graph/create \
  -H "Content-Type: application/json" \
  -d '{"name":"simple","start_node":"start","nodes":[{"name":"start","type":"start"},{"name":"increment","type":"standard","function":"increment"},{"name":"end","type":"end"}],"edges":[{"source":"start","target":"increment"},{"source":"increment","target":"end"}]}'
```

- Run synchronously:

```bash
curl -X POST http://localhost:8000/graph/run \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"<graph_id>", "initial_state":{"value":1}}'
```

- Run asynchronously (background) and poll status/state:

```bash
curl -X POST http://localhost:8000/graph/run_async -H "Content-Type: application/json" -d '{"graph_id":"<graph_id>","initial_state":{}}'
# -> {"run_id":"...","status":"running"}
curl http://localhost:8000/graph/status/<run_id>
curl http://localhost:8000/graph/state/<run_id>
```

- Cancel a background run:

```bash
curl -X POST http://localhost:8000/graph/cancel/<run_id>
```

WebSocket logs

Connect to `ws://localhost:8000/graph/ws/{run_id}` to receive real-time JSON log messages for a run. Example using `websocat`:

```bash
websocat ws://localhost:8000/graph/ws/<run_id>
```

Security Notes

- Dynamic registration executes user-provided Python code. The project includes an AST-based check and limits builtins, and node functions are executed in a subprocess `RemoteFunction` wrapper for additional isolation. This reduces risk but is NOT a fully secure sandbox for untrusted code. For production or untrusted inputs, run code in a hardened container or sandbox with resource limits.

Persistence & Resilience

- Graphs and run metadata are persisted to a local SQLite DB (`workflow.db`). Background runs submitted while a process is running are stored immediately; on restart the engine rehydrates persisted runs and marks previously-running runs as `interrupted` so their state can be inspected.

What I'd improve with more time
- Run dynamic code in an externally managed worker pool / container runtime with strict CPU/memory/time limits
- Implement automatic resume or job requeueing for interrupted runs
- Add more operational metrics and structured logs

Tests

Run the test suite:

```bash
pytest
```
| POST | `/graph/run_async` | Start a workflow in background, returns `run_id` |
| GET | `/graph/state/{run_id}` | Get current state of a workflow run |
| GET | `/graph/list` | List all registered graphs |
| GET | `/graph/{graph_id}` | Get details of a specific graph |
| GET | `/graph/runs` | List all workflow runs |

### Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tools/list` | List all registered tools |
| POST | `/tools/register` | Dynamically register a new tool function |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| WS `/graph/ws/{run_id}` | Stream real-time execution logs |

## Sample Workflow: Code Review Mini-Agent

The project includes a pre-registered **Code Review Mini-Agent** workflow that demonstrates all engine capabilities:

### Workflow Steps

1. **Extract Functions**: Parse Python code and extract function definitions
2. **Check Complexity**: Calculate cyclomatic and cognitive complexity metrics
3. **Detect Issues**: Identify code smells (long functions, magic numbers, etc.)
4. **Suggest Improvements**: Generate prioritized improvement suggestions
5. **Calculate Score**: Compute overall quality score (0-10)
6. **Loop**: Continue improving until `quality_score >= threshold`

### Usage Example

Run synchronously:
```bash
curl -X POST http://localhost:8000/graph/run \
  -H "Content-Type: application/json" \
  -d '{
    "graph_id": "<graph_id_from_list>",
    "initial_state": {
      "code": "def calculate(x, y):\n    if x > 10:\n        return x*y\n    return x+y",
      "quality_threshold": 6.0
    }
  }'
```

Run in background and poll state:
```bash
# Start
RUN_ID=$(curl -s -X POST http://localhost:8000/graph/run_async \
  -H "Content-Type: application/json" \
  -d '{"graph_id": "<graph_id_from_list>", "initial_state": {"code": "def f():\n    return 1"}}' | jq -r '.run_id')

# Poll state
curl -s http://localhost:8000/graph/state/$RUN_ID
```

### Sample Response

```json
{
  "run_id": "abc123",
  "graph_id": "xyz789",
  "status": "completed",
  "final_state": {
    "run_id": "abc123",
    "data": {
      "functions": [...],
      "complexity_results": [...],
      "issues": [...],
      "suggestions": [...],
      "quality_score": 7.2,
      "improvements_applied": 3
    },
    "execution_log": [...]
  }
}
```

## Creating Custom Workflows

### Via API

```bash
curl -X POST http://localhost:8000/graph/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_workflow",
    "nodes": [
      {"name": "start", "type": "start"},
      {"name": "process", "type": "standard", "function": "my_function"},
      {"name": "end", "type": "end"}
    ],
    "edges": [
      {"source": "start", "target": "process"},
      {"source": "process", "target": "end"}
    ]
  }'
```

### Via Python

```python
from app.core.engine import WorkflowEngine, WorkflowGraph, NodeType
from app.core.state import WorkflowState

# Create a graph
graph = WorkflowGraph(name="my_workflow")

# Define node function
def my_node(state: WorkflowState) -> WorkflowState:
    value = state.get("input", 0)
    state.set("output", value * 2)
    return state

# Add nodes
graph.add_node("start", node_type=NodeType.START)
graph.add_node("process", func=my_node)
graph.add_node("end", node_type=NodeType.END)

# Add edges
graph.add_edge("start", "process")
graph.add_edge("process", "end")

# Run the workflow
engine = WorkflowEngine()
engine.register_graph(graph)

import asyncio
result = asyncio.run(engine.run(
    graph_id=graph.graph_id,
    initial_state={"input": 5}
))
print(result["final_state"]["data"]["output"])  # 10
```

### Creating Loop Workflows

```python
def should_continue(state: WorkflowState) -> bool:
    return state.get("counter", 0) < 5

def increment(state: WorkflowState) -> WorkflowState:
    state.set("counter", state.get("counter", 0) + 1)
    return state

graph.add_node(
    "loop_node",
    func=increment,
    node_type=NodeType.LOOP,
    condition=should_continue,
    max_iterations=10
)
```

## Tool Registry

Register custom tools that nodes can invoke:

```python
from app.core.tools import tool, get_global_registry

@tool(name="my_tool", description="Does something useful")
def my_tool_function(arg1: str, arg2: int) -> dict:
    return {"result": f"{arg1} - {arg2}"}

# Or programmatically
registry = get_global_registry()
registry.add(my_function, name="another_tool")

# Invoke tools
result = registry.invoke("my_tool", "hello", 42)
```

## WebSocket Log Streaming

Connect to the WebSocket endpoint to receive real-time execution updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/graph/ws/{run_id}');

ws.onmessage = (event) => {
  const logEntry = JSON.parse(event.data);
  console.log(`[${logEntry.node}] ${logEntry.status}: ${logEntry.message}`);
};

// Keep alive
setInterval(() => ws.send('ping'), 30000);
```

## What I Would Improve With More Time

### Core Engine

1. **Persistent Storage**: Add SQLite/PostgreSQL backend for graphs and runs
2. **Parallel Node Execution**: Execute independent nodes concurrently
3. **Subgraph Support**: Allow nesting graphs within graphs
4. **Checkpointing**: Save and restore workflow state for long-running processes
5. **Event-Driven Nodes**: Support for waiting on external events

### API & Infrastructure

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Protect against abuse
3. **Background Task Queue**: Use Celery/RQ for long-running workflows
4. **Metrics & Monitoring**: Prometheus metrics, distributed tracing
5. **Graph Visualization**: API endpoint to generate workflow diagrams

### Developer Experience

1. **CLI Tool**: Command-line interface for workflow management
2. **Graph DSL**: Domain-specific language for defining workflows
3. **Type-Safe State**: Generic typed state models per workflow
4. **Testing Utilities**: Helpers for unit testing nodes and workflows
5. **Hot Reload**: Update node functions without restarting

### Code Review Workflow

1. **Real Code Transformations**: Actually apply refactoring suggestions
2. **AST-Based Analysis**: Deeper code analysis using AST
3. **Multiple Language Support**: Extend beyond Python
4. **Integration with Linters**: Connect to pylint, flake8, etc.
5. **Git Integration**: Analyze changed files in PRs

## License

MIT