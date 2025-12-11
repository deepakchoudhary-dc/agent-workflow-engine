"""
Code Review Mini-Agent Workflow.

A sample workflow demonstrating the workflow engine capabilities:
1. Extract functions from code
2. Check complexity of each function
3. Detect basic issues (code smells)
4. Suggest improvements
5. Loop until quality_score >= threshold

This workflow is rule-based and does not require any ML models.
"""

from __future__ import annotations

import re
import ast
from typing import Any, Dict, List

from ..core.state import WorkflowState
from ..core.engine import WorkflowGraph, NodeType
from ..core.tools import tool


# ============================================================================
# Tools for Code Analysis
# ============================================================================


@tool(
    name="extract_functions",
    description="Extract function definitions from Python code",
)
def extract_functions(code: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from Python code.

    Returns a list of dictionaries containing function info:
    - name: Function name
    - args: List of argument names
    - body: Function body as string
    - lineno: Starting line number
    - lines: Number of lines in function
    """
    functions = []

    try:
        tree = ast.parse(code)
        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function body
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else start_line + 10
                )
                body_lines = lines[start_line:end_line]

                functions.append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "body": "\n".join(body_lines),
                        "lineno": node.lineno,
                        "lines": end_line - start_line,
                    }
                )
    except SyntaxError:
        # Fallback to regex for invalid Python
        pattern = r"def\s+(\w+)\s*\([^)]*\):"
        for match in re.finditer(pattern, code):
            functions.append(
                {
                    "name": match.group(1),
                    "args": [],
                    "body": "",
                    "lineno": code[: match.start()].count("\n") + 1,
                    "lines": 1,
                }
            )

    return functions


@tool(
    name="calculate_complexity",
    description="Calculate cyclomatic complexity of a function",
)
def calculate_complexity(func_body: str) -> Dict[str, Any]:
    """
    Calculate complexity metrics for a function.

    Returns:
    - cyclomatic: Cyclomatic complexity (based on control flow)
    - cognitive: Simplified cognitive complexity
    - lines: Number of lines
    """
    # Count control flow statements for cyclomatic complexity
    control_patterns = [
        r"\bif\b",
        r"\belif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\band\b",
        r"\bor\b",
        r"\bexcept\b",
        r"\bwith\b",
    ]

    cyclomatic = 1  # Base complexity
    for pattern in control_patterns:
        cyclomatic += len(re.findall(pattern, func_body))

    # Simplified cognitive complexity
    nesting_patterns = [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\btry\b"]
    cognitive = 0
    nesting_level = 0

    for line in func_body.split("\n"):
        indent = len(line) - len(line.lstrip())
        nesting_level = indent // 4

        for pattern in nesting_patterns:
            if re.search(pattern, line):
                cognitive += 1 + nesting_level

    lines = len([ln for ln in func_body.split("\n") if ln.strip()])

    return {
        "cyclomatic": cyclomatic,
        "cognitive": cognitive,
        "lines": lines,
    }


@tool(name="detect_issues", description="Detect code issues and smells")
def detect_issues(func_body: str, func_name: str) -> List[Dict[str, str]]:
    """
    Detect common code issues and smells.

    Returns a list of issues found:
    - type: Issue type
    - message: Description
    - severity: low, medium, high
    """
    issues = []

    # Check function length
    lines = len([ln for ln in func_body.split("\n") if ln.strip()])
    if lines > 30:
        issues.append(
            {
                "type": "long_function",
                "message": f"Function '{func_name}' is too long ({lines} lines). Consider breaking it up.",
                "severity": "medium",
            }
        )

    # Check for magic numbers
    magic_numbers = re.findall(r"\b(?<!\.)\d{2,}\b", func_body)
    if magic_numbers:
        issues.append(
            {
                "type": "magic_numbers",
                "message": f"Magic numbers found: {magic_numbers[:3]}. Consider using named constants.",
                "severity": "low",
            }
        )

    # Check for too many arguments (from function signature)
    arg_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", func_body)
    if arg_match:
        args = [a.strip() for a in arg_match.group(1).split(",") if a.strip()]
        if len(args) > 5:
            issues.append(
                {
                    "type": "too_many_args",
                    "message": f"Function has {len(args)} arguments. Consider using a config object.",
                    "severity": "medium",
                }
            )

    # Check for nested loops
    nested_loop = re.search(r"for\s+.*:\s*\n\s+.*for\s+", func_body)
    if nested_loop:
        issues.append(
            {
                "type": "nested_loops",
                "message": "Nested loops detected. Consider refactoring for better readability.",
                "severity": "medium",
            }
        )

    # Check for broad exception handling
    if re.search(r"except\s*:", func_body) or re.search(
        r"except\s+Exception\s*:", func_body
    ):
        issues.append(
            {
                "type": "broad_exception",
                "message": "Broad exception handling. Consider catching specific exceptions.",
                "severity": "medium",
            }
        )

    # Check for TODO/FIXME comments
    todos = re.findall(r"#\s*(TODO|FIXME|XXX|HACK)", func_body, re.IGNORECASE)
    if todos:
        issues.append(
            {
                "type": "unresolved_todos",
                "message": f"Found {len(todos)} TODO/FIXME comments that need attention.",
                "severity": "low",
            }
        )

    # Check for print statements (should use logging)
    prints = re.findall(r"\bprint\s*\(", func_body)
    if prints:
        issues.append(
            {
                "type": "print_statements",
                "message": f"Found {len(prints)} print statements. Consider using logging.",
                "severity": "low",
            }
        )

    # Check for hardcoded strings
    hardcoded = re.findall(r'["\'][^"\']{20,}["\']', func_body)
    if len(hardcoded) > 2:
        issues.append(
            {
                "type": "hardcoded_strings",
                "message": "Multiple hardcoded strings. Consider using constants or config.",
                "severity": "low",
            }
        )

    return issues


@tool(name="suggest_improvements", description="Suggest improvements based on issues")
def suggest_improvements(issues: List[Dict], complexity: Dict) -> List[Dict[str, str]]:
    """
    Generate improvement suggestions based on detected issues and complexity.

    Returns a list of suggestions:
    - category: Improvement category
    - suggestion: Description of improvement
    - priority: 1-5 (1 = highest)
    """
    suggestions = []

    # Complexity-based suggestions
    if complexity.get("cyclomatic", 0) > 10:
        suggestions.append(
            {
                "category": "complexity",
                "suggestion": "High cyclomatic complexity. Extract helper functions to reduce branches.",
                "priority": 1,
            }
        )

    if complexity.get("cognitive", 0) > 15:
        suggestions.append(
            {
                "category": "readability",
                "suggestion": "High cognitive complexity. Flatten nested structures and use early returns.",
                "priority": 2,
            }
        )

    if complexity.get("lines", 0) > 50:
        suggestions.append(
            {
                "category": "structure",
                "suggestion": "Function is very long. Apply Single Responsibility Principle.",
                "priority": 1,
            }
        )

    # Issue-based suggestions
    issue_types = {issue["type"] for issue in issues}

    if "long_function" in issue_types:
        suggestions.append(
            {
                "category": "refactoring",
                "suggestion": "Extract logical sections into separate functions with descriptive names.",
                "priority": 2,
            }
        )

    if "magic_numbers" in issue_types:
        suggestions.append(
            {
                "category": "maintainability",
                "suggestion": "Replace magic numbers with named constants (e.g., MAX_RETRIES = 3).",
                "priority": 3,
            }
        )

    if "too_many_args" in issue_types:
        suggestions.append(
            {
                "category": "design",
                "suggestion": "Consider using a dataclass or NamedTuple for function parameters.",
                "priority": 2,
            }
        )

    if "nested_loops" in issue_types:
        suggestions.append(
            {
                "category": "performance",
                "suggestion": "Consider using list comprehensions, itertools, or extracting inner loops.",
                "priority": 2,
            }
        )

    if "broad_exception" in issue_types:
        suggestions.append(
            {
                "category": "error_handling",
                "suggestion": "Catch specific exceptions and add meaningful error messages.",
                "priority": 2,
            }
        )

    if "print_statements" in issue_types:
        suggestions.append(
            {
                "category": "logging",
                "suggestion": "Replace print statements with proper logging using the logging module.",
                "priority": 3,
            }
        )

    # Sort by priority
    suggestions.sort(key=lambda x: x["priority"])

    return suggestions


@tool(name="calculate_quality_score", description="Calculate overall quality score")
def calculate_quality_score(
    issues: List[Dict], complexity: Dict, improvements_applied: int = 0
) -> float:
    """
    Calculate a quality score from 0-10 based on issues and complexity.

    Higher score = better quality.
    """
    score = 10.0

    # Deduct for complexity
    cyclomatic = complexity.get("cyclomatic", 0)
    if cyclomatic > 5:
        score -= min((cyclomatic - 5) * 0.3, 2.0)

    cognitive = complexity.get("cognitive", 0)
    if cognitive > 10:
        score -= min((cognitive - 10) * 0.2, 2.0)

    # Deduct for issues
    severity_penalties = {"high": 1.0, "medium": 0.5, "low": 0.2}
    for issue in issues:
        score -= severity_penalties.get(issue.get("severity", "low"), 0.2)

    # Bonus for applied improvements
    score += improvements_applied * 0.5

    # Clamp to 0-10
    return max(0.0, min(10.0, score))


# ============================================================================
# Workflow Node Functions
# ============================================================================


def extract_functions_node(state: WorkflowState) -> WorkflowState:
    """Node: Extract functions from the input code."""
    code = state.get("code", "")

    if not code:
        state.set("error", "No code provided")
        state.set("functions", [])
        return state

    functions = extract_functions(code)
    state.set("functions", functions)
    state.set("function_count", len(functions))

    return state


def check_complexity_node(state: WorkflowState) -> WorkflowState:
    """Node: Calculate complexity for each function."""
    functions = state.get("functions", [])

    complexity_results = []
    total_complexity = 0

    for func in functions:
        complexity = calculate_complexity(func.get("body", ""))
        complexity_results.append({"function": func["name"], **complexity})
        total_complexity += complexity["cyclomatic"]

    state.set("complexity_results", complexity_results)
    state.set("total_complexity", total_complexity)
    state.set("avg_complexity", total_complexity / len(functions) if functions else 0)

    return state


def detect_issues_node(state: WorkflowState) -> WorkflowState:
    """Node: Detect issues in each function."""
    functions = state.get("functions", [])

    all_issues = []
    for func in functions:
        issues = detect_issues(func.get("body", ""), func.get("name", "unknown"))
        for issue in issues:
            issue["function"] = func["name"]
            all_issues.append(issue)

    state.set("issues", all_issues)
    state.set("issue_count", len(all_issues))

    # Count by severity
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for issue in all_issues:
        severity = issue.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    state.set("severity_counts", severity_counts)

    return state


def suggest_improvements_node(state: WorkflowState) -> WorkflowState:
    """Node: Generate improvement suggestions."""
    issues = state.get("issues", [])
    complexity_results = state.get("complexity_results", [])

    # Aggregate complexity
    avg_complexity = {
        "cyclomatic": sum(c.get("cyclomatic", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
        "cognitive": sum(c.get("cognitive", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
        "lines": sum(c.get("lines", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
    }

    suggestions = suggest_improvements(issues, avg_complexity)

    state.set("suggestions", suggestions)
    state.set("suggestion_count", len(suggestions))

    return state


def calculate_score_node(state: WorkflowState) -> WorkflowState:
    """Node: Calculate the overall quality score."""
    issues = state.get("issues", [])
    complexity_results = state.get("complexity_results", [])
    improvements_applied = state.get("improvements_applied", 0)

    # Aggregate complexity
    avg_complexity = {
        "cyclomatic": sum(c.get("cyclomatic", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
        "cognitive": sum(c.get("cognitive", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
        "lines": sum(c.get("lines", 0) for c in complexity_results)
        / max(len(complexity_results), 1),
    }

    score = calculate_quality_score(issues, avg_complexity, improvements_applied)

    state.set("quality_score", score)
    state.set("previous_scores", state.get("previous_scores", []) + [score])

    return state


def apply_improvements_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Simulate applying improvements.

    In a real system, this would apply actual code transformations.
    Here we simulate by reducing issue count and increasing improvements_applied.
    """
    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])

    # Simulate applying one improvement per iteration
    if suggestions and issues:
        # Remove lowest severity issues (simulating fixes)
        issues = sorted(
            issues,
            key=lambda x: {"low": 0, "medium": 1, "high": 2}.get(
                x.get("severity", "low"), 0
            ),
        )
        if issues:
            # Remove one issue
            issues.pop(0)
            state.set("issues", issues)
            state.set("issue_count", len(issues))

        # Track improvements applied
        state.set("improvements_applied", state.get("improvements_applied", 0) + 1)

    return state


def should_continue_loop(state: WorkflowState) -> bool:
    """Condition: Check if we should continue the improvement loop."""
    quality_score = state.get("quality_score", 0)
    threshold = state.get("quality_threshold", 7.0)
    max_iterations = state.get("max_iterations", 5)
    improvements_applied = state.get("improvements_applied", 0)

    # Continue if score below threshold and we haven't exceeded max iterations
    return quality_score < threshold and improvements_applied < max_iterations


# ============================================================================
# Workflow Graph Builder
# ============================================================================


def create_code_review_workflow() -> WorkflowGraph:
    """
    Create the Code Review Mini-Agent workflow graph.

    Flow:
    start -> extract_functions -> check_complexity -> detect_issues ->
    suggest_improvements -> calculate_score -> [loop decision] ->
        if score < threshold: apply_improvements -> calculate_score
        else: end
    """
    graph = WorkflowGraph(name="code_review_agent")

    # Add nodes
    graph.add_node(
        name="start",
        node_type=NodeType.START,
    )

    graph.add_node(
        name="extract_functions",
        func=extract_functions_node,
        node_type=NodeType.STANDARD,
    )

    graph.add_node(
        name="check_complexity",
        func=check_complexity_node,
        node_type=NodeType.STANDARD,
    )

    graph.add_node(
        name="detect_issues",
        func=detect_issues_node,
        node_type=NodeType.STANDARD,
    )

    graph.add_node(
        name="suggest_improvements",
        func=suggest_improvements_node,
        node_type=NodeType.STANDARD,
    )

    graph.add_node(
        name="calculate_score",
        func=calculate_score_node,
        node_type=NodeType.LOOP,
        condition=should_continue_loop,
        max_iterations=10,
    )

    graph.add_node(
        name="apply_improvements",
        func=apply_improvements_node,
        node_type=NodeType.STANDARD,
    )

    graph.add_node(
        name="end",
        node_type=NodeType.END,
    )

    # Add edges
    graph.add_edge("start", "extract_functions")
    graph.add_edge("extract_functions", "check_complexity")
    graph.add_edge("check_complexity", "detect_issues")
    graph.add_edge("detect_issues", "suggest_improvements")
    graph.add_edge("suggest_improvements", "calculate_score")
    graph.add_edge("calculate_score", "apply_improvements", label="loop")  # Loop back
    graph.add_edge("calculate_score", "end")  # Exit when condition fails
    graph.add_edge("apply_improvements", "calculate_score")  # Continue loop

    return graph


# ============================================================================
# Pre-registered workflow for easy API usage
# ============================================================================

# Register node functions for API-based graph creation
from ..api.routes import register_node_function  # noqa: E402


def register_code_review_functions():
    """Register all code review functions for API use."""
    register_node_function("extract_functions", extract_functions_node)
    register_node_function("check_complexity", check_complexity_node)
    register_node_function("detect_issues", detect_issues_node)
    register_node_function("suggest_improvements", suggest_improvements_node)
    register_node_function("calculate_score", calculate_score_node)
    register_node_function("apply_improvements", apply_improvements_node)
    register_node_function("should_continue_loop", should_continue_loop)
