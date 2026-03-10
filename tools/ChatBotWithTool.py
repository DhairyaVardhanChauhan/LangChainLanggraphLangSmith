import os

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
import json

# ---- RICH UI ----
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich import box

console = Console()
load_dotenv()
api_key = os.getenv("EXCHANGE_API_KEY")
@tool
def multiply(a: int, b: int) -> int:
    """Used to multiply two numbers"""
    return a * b

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}'

  response = requests.get(url)
  # print(response.json())
  print(response.json())
  return response.json()

get_conversion_factor.invoke({"base_currency": "INR", "target_currency": "USD"})
llm_client = ChatAnthropic(model_name="claude-sonnet-4-6")
llm_with_tools = llm_client.bind_tools([multiply, get_conversion_factor])

chat_history = []

console.print(
    Panel.fit(
        "[bold cyan]🔢 LangChain Tool Debug UI[/bold cyan]\nType 'exit' to quit",
        border_style="cyan",
        box=box.ROUNDED,
    )
)


#  prompt what is 9 *10*100*1000*1223
while True:
    query = console.input("\n[bold green]You:[/bold green] ")

    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    while True:
        response = llm_with_tools.invoke(chat_history)
        if not response.tool_calls:
            console.print(
                Panel(
                    response.content,
                    title="[bold blue]Assistant[/bold blue]",
                    border_style="blue",
                )
            )
            break
        chat_history.append(response)
        print(response.tool_calls)
        for call in response.tool_calls:
            console.print(
                Panel(
                    Pretty({
                        "Tool Name": call["name"],
                        "Arguments": call["args"],
                        "Tool Call ID": call["id"]
                    }),
                    title="[bold yellow]🔧 TOOL USE DETECTED[/bold yellow]",
                    border_style="yellow",
                )
            )
            result = None
            if call["name"] == "multiply":
                result = multiply.invoke(call)

            elif call["name"] == "get_conversion_factor":
                result = get_conversion_factor.invoke(call)
            console.print(
                Panel(
                    Pretty({
                        "Result": result,
                        "Returned To ID": call["id"]
                    }),
                    title="[bold magenta]✅ TOOL RESULT[/bold magenta]",
                    border_style="magenta",
                )
            )
            chat_history.append(result)
# if tool2 depends on output of tool1 use
# def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:-