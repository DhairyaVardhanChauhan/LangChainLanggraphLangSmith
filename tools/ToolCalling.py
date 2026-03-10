from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty


load_dotenv()

console = Console()
@tool
def multiply(a: int, b: int) -> int:
    """Used to multiply two numbers"""
    return a * b

llm_client = ChatAnthropic(model_name="claude-sonnet-4-6")
llm_with_tools = llm_client.bind_tools([multiply])

response  = llm_with_tools.invoke("Multiply 10 with 12 and the result with 30")
# ----------------------------
# Pretty Print Response
# ----------------------------

console.print(Panel("Raw AIMessage", style="bold cyan"))
console.print(Pretty(response))

# If tool call exists, display nicely
if response.tool_calls:
    console.print(Panel("Tool Call Requested", style="bold green"))
    console.print(Pretty(response.tool_calls))