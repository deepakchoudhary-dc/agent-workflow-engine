"""Subprocess-based sandbox runner for user-provided node functions.

This provides a best-effort isolation: each call runs in a fresh Python
subprocess with a tiny driver that exposes a limited `State` wrapper with
`.get()` and `.set()` helpers. This is not bulletproof but reduces the
impact of executing untrusted code in-process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from threading import Lock


_DRIVER_TEMPLATE = r"""
<<USER_CODE>>

import sys, json, time
try:
    import resource
    # Apply resource limits (placeholders will be replaced)
    resource.setrlimit(resource.RLIMIT_AS, (<<MEM_LIMIT>>, <<MEM_LIMIT>>))
    resource.setrlimit(resource.RLIMIT_CPU, (<<CPU_LIMIT>>, <<CPU_LIMIT>>))
except Exception:
    pass

class State:
    def __init__(self, data):
        self.data = data
    def get(self, k, default=None):
        return self.data.get(k, default)
    def set(self, k, v):
        self.data[k] = v
    def snapshot(self):
        return self.data

data = json.loads(sys.stdin.read() or "{}")
state = State(data)
try:
    result = <<FN_NAME>>(state)
except Exception as e:
    print(json.dumps({"__error__": str(e)}))
    sys.exit(2)

print(json.dumps(state.data))
"""


def _run_subprocess(
    code: str,
    fn_name: str,
    input_data: Dict[str, Any],
    timeout: Optional[float],
    cpu_limit: int,
    mem_limit: int,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    driver = (
        _DRIVER_TEMPLATE.replace("<<USER_CODE>>", code)
        .replace("<<FN_NAME>>", fn_name)
        .replace("<<CPU_LIMIT>>", str(int(cpu_limit)))
        .replace("<<MEM_LIMIT>>", str(int(mem_limit)))
    )
    # Run via Popen so we can track the child PID and support cooperative
    # cancellation. Register the PID against the optional run_id while running.
    proc = subprocess.Popen(
        [sys.executable, "-c", driver],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Register PID for cancellation tracking
    if run_id:
        with _PID_LOCK:
            _ACTIVE_PIDS.setdefault(run_id, set()).add(proc.pid)

    try:
        try:
            out_bytes, err_bytes = proc.communicate(
                input=json.dumps(input_data).encode(), timeout=timeout
            )
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            raise RuntimeError("Subprocess timed out")

        returncode = proc.returncode

        if returncode != 0:
            # Try to decode error payload
            try:
                payload = json.loads((out_bytes or err_bytes).decode())
                if isinstance(payload, dict) and payload.get("__error__"):
                    raise RuntimeError(payload.get("__error__"))
            except Exception:
                raise RuntimeError(
                    f"Subprocess failed: {returncode} stderr={(err_bytes or b'').decode()}"
                )

        try:
            return json.loads((out_bytes or b"").decode())
        except Exception as e:
            stdout_text = (out_bytes or b"").decode()
            stderr_text = (err_bytes or b"").decode()
            raise RuntimeError(
                "Failed to decode subprocess output: "
                + str(e)
                + "\nstdout="
                + stdout_text
                + "\nstderr="
                + stderr_text
            )
    finally:
        if run_id:
            with _PID_LOCK:
                s = _ACTIVE_PIDS.get(run_id)
                if s and proc.pid in s:
                    s.discard(proc.pid)
                if s and len(s) == 0:
                    _ACTIVE_PIDS.pop(run_id, None)


@dataclass
class RemoteFunction:
    """Callable wrapper that executes user code in a subprocess on each call.

    Adds per-call CPU and memory caps (Linux only, uses `resource` inside the
    subprocess). Defaults are modest but configurable when constructing the
    wrapper.
    """

    code: str
    name: str
    timeout: Optional[float] = 5.0
    cpu_seconds: int = 2
    mem_bytes: int = 128 * 1024 * 1024

    def __call__(self, state) -> Any:
        # state is a WorkflowState; we pass only its data mapping to subprocess
        data = (
            state.to_output()["data"]
            if hasattr(state, "to_output")
            else getattr(state, "data", {})
        )
        run_id = getattr(state, "run_id", None)
        out = _run_subprocess(
            self.code,
            self.name,
            data,
            self.timeout,
            self.cpu_seconds,
            self.mem_bytes,
            run_id=run_id,
        )
        if not isinstance(out, dict):
            raise RuntimeError("Remote function did not return a dict-like state")
        # update original state's data in-place and return state
        if hasattr(state, "update"):
            state.update(out)
        else:
            state.data = out
        return state


# Active subprocess tracking for cooperative cancellation
_ACTIVE_PIDS: Dict[str, set] = {}
_PID_LOCK = Lock()


def cancel_run(run_id: str, grace: float = 0.5) -> int:
    """Attempt to terminate subprocesses associated with a run_id.

    Returns the number of PIDs signalled.
    """
    with _PID_LOCK:
        pids = set(_ACTIVE_PIDS.get(run_id, set()))
    signalled = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            signalled += 1
        except ProcessLookupError:
            # Already gone
            with _PID_LOCK:
                s = _ACTIVE_PIDS.get(run_id)
                if s and pid in s:
                    s.discard(pid)
    # Give processes a short grace period, then force kill remaining
    time.sleep(grace)
    with _PID_LOCK:
        remaining = set(_ACTIVE_PIDS.get(run_id, set()))
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            signalled += 1
        except ProcessLookupError:
            pass
    # Clean up
    with _PID_LOCK:
        _ACTIVE_PIDS.pop(run_id, None)
    return signalled
