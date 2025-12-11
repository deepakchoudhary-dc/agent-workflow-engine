"""
Tool Registry.

Provides a registry for tools (Python functions) that can be called by workflow nodes.
Supports both decorator-based and programmatic registration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from functools import wraps
import inspect


class ToolRegistry:
    """
    A registry for workflow tools.

    Tools are Python functions that can be invoked by nodes during workflow execution.
    Each tool is registered with a unique name and optional metadata.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> Callable:
        """
        Decorator to register a function as a tool.

        Args:
            name: Optional name for the tool. Defaults to function name.
            description: Optional description. Defaults to function docstring.

        Example:
            @registry.register(name="my_tool")
            def my_function(arg1, arg2):
                return result
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            # Extract function signature
            sig = inspect.signature(func)
            params = {
                param.name: {
                    "annotation": (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    ),
                    "default": (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    "required": param.default == inspect.Parameter.empty,
                }
                for param in sig.parameters.values()
            }

            self._tools[tool_name] = {
                "function": func,
                "description": tool_desc.strip(),
                "parameters": params,
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def add(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Programmatically register a function as a tool.

        Args:
            func: The function to register.
            name: Optional name for the tool. Defaults to function name.
            description: Optional description. Defaults to function docstring.
        """
        decorator = self.register(name=name, description=description)
        decorator(func)

    def get(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        tool = self._tools.get(name)
        return tool["function"] if tool else None

    def invoke(self, name: str, *args, **kwargs) -> Any:
        """
        Invoke a tool by name.

        Args:
            name: The tool name.
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            The tool's return value.

        Raises:
            KeyError: If the tool is not found.
        """
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")
        return tool["function"](*args, **kwargs)

    def list_tools(self) -> list[Dict[str, Any]]:
        """List all registered tools with their metadata."""
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in self._tools.items()
        ]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global tool registry instance
_global_registry = ToolRegistry()


def tool(name: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to register a function as a tool in the global registry.

    Example:
        @tool(name="detect_smells")
        def detect_code_smells(code: str) -> dict:
            return {"issues": 3}
    """
    return _global_registry.register(name=name, description=description)


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry
