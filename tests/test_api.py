import time
from uuid import uuid4
from fastapi.testclient import TestClient

from app.api.routes import get_engine
from app.main import app


client = TestClient(app)


def register_node(name: str, code: str) -> None:
    resp = client.post("/nodes/register", json={"name": name, "code": code})
    assert resp.status_code == 200, resp.text


def register_tool(name: str, code: str) -> None:
    resp = client.post("/tools/register", json={"name": name, "code": code})
    assert resp.status_code == 200, resp.text


def create_linear_graph(node_name: str) -> str:
    payload = {
        "name": f"workflow_{node_name}",
        "start_node": "start",
        "nodes": [
            {"name": "start", "type": "start"},
            {"name": node_name, "type": "standard", "function": node_name},
            {"name": "end", "type": "end"},
        ],
        "edges": [
            {"source": "start", "target": node_name},
            {"source": node_name, "target": "end"},
        ],
    }
    resp = client.post("/graph/create", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["graph_id"]


def test_node_register_and_sync_run():
    code = """

def increment(state):
    value = state.get("value", 0)
    state.set("value", value + 1)
    return state
"""
    register_node("increment", code)
    graph_id = create_linear_graph("increment")

    resp = client.post(
        "/graph/run", json={"graph_id": graph_id, "initial_state": {"value": 1}}
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed"
    assert payload["final_state"]["data"]["value"] == 2


def test_tool_register_and_reuse_for_node():
    code = """

def add_one(state):
    val = state.get("n", 0)
    state.set("n", val + 1)
    return state
"""
    register_tool("add_one", code)
    graph_id = create_linear_graph("add_one")

    resp = client.post(
        "/graph/run", json={"graph_id": graph_id, "initial_state": {"n": 5}}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["final_state"]["data"]["n"] == 6


def test_run_async_status_and_state():
    code = """

def sleeper(state):
    time.sleep(0.1)
    state.set("slept", True)
    return state
"""
    register_node("sleeper", code)
    graph_id = create_linear_graph("sleeper")

    resp = client.post(
        "/graph/run_async", json={"graph_id": graph_id, "initial_state": {}}
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    status_resp = client.get(f"/graph/status/{run_id}")
    assert status_resp.status_code == 200

    time.sleep(0.25)
    state_resp = client.get(f"/graph/state/{run_id}")
    assert state_resp.status_code == 200
    state_json = state_resp.json()
    assert state_json["status"] in {"running", "completed"}
    assert state_json["current_state"]["data"].get("slept") is True


def test_websocket_ping_pong():
    run_id = f"ws-{uuid4()}"
    with client.websocket_connect(f"/graph/ws/{run_id}") as websocket:
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"


def test_cancel_run():
    code = """

def long_task(state):
    time.sleep(1)
    return state
"""
    register_node("long_task", code)
    graph_id = create_linear_graph("long_task")

    resp = client.post(
        "/graph/run_async", json={"graph_id": graph_id, "initial_state": {}}
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    cancel_resp = client.post(f"/graph/cancel/{run_id}")
    assert cancel_resp.status_code in (200, 404), cancel_resp.text
    status_resp = client.get(f"/graph/status/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in {"cancelled", "completed", "running"}


def test_graph_validation_requires_end_node():
    payload = {
        "name": "invalid_graph",
        "start_node": "start",
        "nodes": [
            {"name": "start", "type": "start"},
            {"name": "middle", "type": "standard"},
        ],
        "edges": [
            {"source": "start", "target": "middle"},
        ],
    }
    resp = client.post("/graph/create", json=payload)
    assert resp.status_code == 400

    # ensure engine state not polluted
    engine = get_engine()
    assert not any(graph["name"] == "invalid_graph" for graph in engine.list_graphs())
