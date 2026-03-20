import requests
import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun


# ==============================
# Environment Setup
# ==============================

load_dotenv()

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


# ==============================
# LLM Setup
# ==============================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ==============================
# Tools
# ==============================

search_tool = DuckDuckGoSearchRun()


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform math operations on two numbers.

    Supported operations:
    - add
    - subtract
    - multiply
    - divide
    """

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": "Cannot divide by zero" if b == 0 else a / b
    }
    return operations.get(operation, "Invalid operation")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price using Alpha Vantage API.
    Example symbols: AAPL, TSLA, MSFT
    """
    print("Fetching latest stock price for " + symbol)
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=OOCIBMBBTXN9C2CZ"
    )

    response = requests.get(url)
    return response.json()


tools = [search_tool, calculator, get_stock_price]


# ==============================
# LLM + Tools Binding
# ==============================

llm_with_tools = llm.bind_tools(
    tools,
    tool_choice="auto",
    parallel_tool_calls=False
)

tool_node = ToolNode(tools)


# ==============================
# Graph State
# ==============================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================
# Chat Node
# ==============================

def chat_node(state: ChatState) -> ChatState:

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    print("\n🤖 AI:", response)

    return {"messages": [response]}


# ==============================
# Graph
# ==============================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ==============================
# Helper
# ==============================

def get_all_threads():
    threads = set()

    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(threads)


# ==============================
# Run Chat
# ==============================

# config = {
#     "configurable": {
#         "thread_id": "2"
#     }
# }
#
# for message_chunk, metadata in chatbot.stream(
#         {"messages": [HumanMessage(content="Hi I am Dhairya!")]},
#         config=config,
#         stream_mode="messages"
# ):
#
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)