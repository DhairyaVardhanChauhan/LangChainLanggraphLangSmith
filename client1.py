import asyncio
from pprint import pprint

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

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
    "manim-server": {
        "command": "/Users/salescode/projects/charjan/.venv/bin/python3",
        "args": [
            "/Users/salescode/downloads/manim-mcp-server/src/manim_server.py"
        ],
        "env": {
            "MANIM_EXECUTABLE": "/Users/salescode/venv/bin/manim"
        },
        "transport": "stdio",
    },

}


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke("crate a video of a boy jumping")
    pprint(response.tool_calls[0])
    selected_tool = response.tool_calls[0]["name"]
    selected_tool_args = response.tool_calls[0]["args"]
    tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
    print(tool_result)
    tool_call_id = response.tool_calls[0]["id"]

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call_id
    )
    final_res = await llm_with_tools.ainvoke(["crate a video of a boy jumping",response,tool_message])
    print(final_res.content)

if __name__ == "__main__":
    asyncio.run(main())