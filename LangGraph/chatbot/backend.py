from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import HumanMessage
import sqlite3

conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)
load_dotenv()
checkpointer = SqliteSaver(conn)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
config = {'configurable':{"thread_id":"2"}}
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
def chat_node(state: ChatState) -> ChatState:
    print("Reached chat state")
    messages = state["messages"]
    response = llm.invoke(messages)
    print(response)
    return {"messages": [response]}


def get_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

graph = StateGraph(ChatState)

graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node",END)
chatbot = graph.compile(checkpointer=checkpointer)

# thread_id = '1'
# while True:
#     inp = input("Enter message: ")
#
#     if inp.lower() == "quit":
#         break
#     config = {'configurable':{"thread_id":thread_id}}
#     ai_response = res.invoke({"messages": [HumanMessage(content=inp)]},config=config)
#     print(ai_response["messages"][-1].content)

for message_chunk, metadata in chatbot.stream({"messages": [HumanMessage(content="Hi i am Dhairya!")]},config=config,stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content, end=" ", flush=True)