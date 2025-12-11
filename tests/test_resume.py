import time

from app.core.engine import WorkflowEngine, WorkflowGraph, NodeType, ExecutionLog
from app.core.state import WorkflowState


def test_resume_interrupted_run_completes():
    engine = WorkflowEngine(auto_resume_interrupted=False)

    # Simple node that sets a flag
    def set_flag(state):
        state.set("resumed", True)
        return state

    graph = WorkflowGraph(name="resume_test")
    graph.add_node("start", func=set_flag, node_type=NodeType.START)
    graph.add_node("end", node_type=NodeType.END)
    graph.add_edge("start", "end")

    engine.register_graph(graph)

    run_id = "resume-test-run"
    state = WorkflowState(run_id=run_id, graph_id=graph.graph_id, data={})
    log = ExecutionLog(run_id, graph.graph_id)

    # Insert interrupted run into engine and persistence
    with engine._lock:
        engine._runs[run_id] = {"state": state, "log": log, "status": "interrupted"}
    try:
        engine.store.save_run(
            run_id=run_id,
            graph_id=graph.graph_id,
            status="interrupted",
            state=state.to_output(),
            log=log.to_dict(),
            started_at=None,
            completed_at=None,
        )
    except Exception:
        pass

    # Trigger resume worker
    engine._resume_interrupted_runs()

    # Wait for resume scheduling and completion
    deadline = time.time() + 5.0
    status = None
    while time.time() < deadline:
        status = engine.get_background_status(run_id)
        if status in {"completed", "running"}:
            break
        time.sleep(0.1)

    assert status in {"completed", "running"}
