import asyncio
import pytest

from app.core.engine import WorkflowEngine
from app.workflows.code_review import (
    create_code_review_workflow,
    register_code_review_functions,
)


@pytest.fixture(scope="module")
def engine():
    register_code_review_functions()
    eng = WorkflowEngine()
    eng.register_graph(create_code_review_workflow())
    return eng


def test_sync_run(engine):
    result = asyncio.run(
        engine.run(
            graph_id=list(engine._graphs.keys())[0],
            initial_state={"code": "def f(x):\n    return x+1", "quality_threshold": 5},
        )
    )
    assert result["status"] == "completed"
    assert "final_state" in result
    assert result["final_state"]["data"]["functions"]


def test_background_run(engine):
    graph_id = list(engine._graphs.keys())[0]
    run_id = engine.run_background(
        graph_id=graph_id, initial_state={"code": "def g():\n    pass"}
    )
    assert run_id
    # allow task to finish
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0.1))
    loop.close()
    state = engine.get_run_state(run_id)
    assert state is not None
    assert state["status"] in {"running", "completed", "error"}
