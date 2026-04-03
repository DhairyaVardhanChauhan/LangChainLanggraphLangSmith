from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "/Users/salescode/Desktop/expense-tracker-mcp-server/.venv/bin/uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "/Users/salescode/Desktop/expense-tracker-mcp-server/main.py"
        ]
    },
}



# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(llm_with_tools,tools):
    # nodes
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        print("==============================")
        print(response)
        print("==============================")
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    print(tools)
    llm_with_tools = llm.bind_tools(tools)
    chatbot = build_graph(llm_with_tools,tools)
    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Roll a dice")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())