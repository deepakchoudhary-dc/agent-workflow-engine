import time

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def register_node(name: str, code: str) -> None:
    resp = client.post("/nodes/register", json={"name": name, "code": code})
    assert resp.status_code == 200, resp.text


def create_linear_graph(node_name: str) -> str:
    payload = {
        "name": f"cancel_workflow_{node_name}",
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


def test_cancel_long_running_subprocess_run():
    code = """

def very_long(state):
    # long sleep to allow cancellation (time is provided in exec globals)
    time.sleep(10)
    state.set("done", True)
    return state
"""
    register_node("very_long", code)
    graph_id = create_linear_graph("very_long")

    resp = client.post(
        "/graph/run_async",
        json={"graph_id": graph_id, "initial_state": {}},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Give the subprocess a moment to start, then cancel
    time.sleep(0.2)
    cancel_resp = client.post(f"/graph/cancel/{run_id}")
    assert cancel_resp.status_code in (200, 404), cancel_resp.text

    # Poll for up to 5s to observe cancelled/completed status
    deadline = time.time() + 5.0
    status = None
    while time.time() < deadline:
        status_resp = client.get(f"/graph/status/{run_id}")
        assert status_resp.status_code == 200
        status = status_resp.json().get("status")
        if status in {"cancelled", "completed"}:
            break
        time.sleep(0.1)

    assert status in {"cancelled", "completed"}
