"""
FastAPI Main Application Entry Point.

Configures and runs the Agent Workflow Engine API.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

try:
    from swagger_ui_bundle import swagger_ui_3_path
except ImportError:
    swagger_ui_3_path = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

from .api.routes import (  # noqa: E402
    router as graph_router,  # noqa: E402
    tool_router,  # noqa: E402
    node_router,  # noqa: E402
    get_engine,  # noqa: E402
    get_websocket_manager,  # noqa: E402
)
from .workflows.code_review import (  # noqa: E402
    create_code_review_workflow,  # noqa: E402
    register_code_review_functions,  # noqa: E402
)
from .core.tools import get_global_registry  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.

    Sets up the workflow engine and registers sample workflows on startup.
    """
    logger = logging.getLogger("agent-workflow-engine")

    # Startup
    logger.info("Starting Agent Workflow Engine...")

    # Register code review functions for API usage
    register_code_review_functions()

    # Create and register the sample code review workflow
    engine = get_engine()
    code_review_workflow = create_code_review_workflow()
    engine.register_graph(code_review_workflow)

    logger.info(
        "Registered workflow: %s (ID: %s)",
        code_review_workflow.name,
        code_review_workflow.graph_id,
    )

    # Setup WebSocket log streaming
    ws_manager = get_websocket_manager()

    async def broadcast_log(step: dict):
        run_id = step.get("run_id")
        if run_id:
            await ws_manager.broadcast(run_id, step)

    engine.on_log(broadcast_log)

    # List registered tools
    tools = get_global_registry().list_tools()
    logger.info("Registered %s tools: %s", len(tools), [t["name"] for t in tools])

    yield

    # Shutdown
    logger.info("Shutting down Agent Workflow Engine...")


# Create FastAPI application
app = FastAPI(
    title="Agent Workflow Engine",
    description="""
A minimal graph-based workflow execution engine inspired by LangGraph.

## Features

- **Nodes**: Python functions that read and modify shared state
- **Edges**: Define execution flow between nodes
- **Branching**: Conditional routing based on state values
- **Looping**: Repeat nodes until conditions are met
- **Tool Registry**: Register and invoke tools from nodes
- **Real-time Logs**: WebSocket streaming of execution logs

## Sample Workflow

The engine comes with a pre-registered **Code Review Mini-Agent** workflow that:
1. Extracts functions from Python code
2. Analyzes complexity metrics
3. Detects code issues and smells
4. Suggests improvements
5. Loops until quality score meets threshold
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(graph_router)
app.include_router(tool_router)
app.include_router(node_router)

SWAGGER_JS_URL = "/static/swagger-ui-bundle.js" if swagger_ui_3_path else None
SWAGGER_CSS_URL = "/static/swagger-ui.css" if swagger_ui_3_path else None

# Serve Swagger UI assets locally when available, otherwise fall back to CDN defaults
if swagger_ui_3_path:
    app.mount("/static", StaticFiles(directory=swagger_ui_3_path), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # Only pass explicit URLs when available to avoid rendering literal "None"
    kwargs = {
        "openapi_url": app.openapi_url,
        "title": f"{app.title} - Swagger UI",
        "oauth2_redirect_url": app.swagger_ui_oauth2_redirect_url,
    }
    if SWAGGER_JS_URL:
        kwargs["swagger_js_url"] = SWAGGER_JS_URL
    if SWAGGER_CSS_URL:
        kwargs["swagger_css_url"] = SWAGGER_CSS_URL

    return get_swagger_ui_html(**kwargs)


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/", tags=["Health"])
async def root():
    """Redirect root to the interactive API docs (Swagger UI)."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
