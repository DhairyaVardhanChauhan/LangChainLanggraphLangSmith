from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
import json

console = Console()


# -----------------------------
# Tools
# -----------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Used to multiply two numbers"""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Used to add two numbers"""
    return a + b


# -----------------------------
# Utility Inspector
# -----------------------------

def inspect_tool(tool_obj, sample_input: dict):
    result = tool_obj.invoke(sample_input)

    console.print(Panel(
        f"[bold green]Result:[/bold green] {result}",
        title=f"Execution → {tool_obj.name}"
    ))

    console.print(Panel(
        f"[bold]Name:[/bold] {tool_obj.name}\n"
        f"[bold]Description:[/bold] {tool_obj.description}",
        title="Metadata"
    ))

    console.print(Panel(
        Pretty(tool_obj.args),
        title="Arguments"
    ))

    schema = tool_obj.args_schema.model_json_schema()

    console.print(Panel(
        json.dumps(schema, indent=4),
        title="Args Schema (JSON)"
    ))


# -----------------------------
# Toolkit
# -----------------------------

class MathToolKit:
    def __init__(self):
        self.tools = [multiply, add]

    def get_tools(self):
        return self.tools

    def display_tools(self):
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="magenta")

        for tool_obj in self.tools:
            table.add_row(tool_obj.name, tool_obj.description)

        console.print(table)


# -----------------------------
# Usage
# -----------------------------

# Inspect one tool deeply
inspect_tool(multiply, {"a": 3, "b": 4})

# Display toolkit
toolkit = MathToolKit()
toolkit.display_tools()