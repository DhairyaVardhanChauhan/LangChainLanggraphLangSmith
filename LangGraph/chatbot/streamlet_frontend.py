import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from backend import chatbot,get_all_threads


# ******************************************************
# ************** Utility Functions *********************
# ******************************************************

def generateThreadId():
    return str(uuid.uuid4())


def resetChat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = generateThreadId()
    addThread(st.session_state["thread_id"])
    st.rerun()


def addThread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def loadConversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    if state and "messages" in state.values:
        return state.values["messages"]

    return []


# ******************************************************
# ************** Session Initialization ****************
# ******************************************************

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generateThreadId()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# ******************************************************
# ******************* Sidebar UI ***********************
# ******************************************************

addThread(st.session_state["thread_id"])

st.sidebar.title("LangGraph Chatbot")
st.sidebar.header("My Conversation")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id

        messages = loadConversation(thread_id)

        tempMessages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            tempMessages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = tempMessages


if st.sidebar.button("New Chat"):
    resetChat()


# ******************************************************
# **************** LangGraph Config ********************
# ******************************************************

config = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn",
}

# ******************************************************
# **************** Chat History Render *****************
# ******************************************************

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ******************************************************
# ******************** User Input **********************
# ******************************************************

user_input = st.chat_input("Type here...")


if user_input:

    # ************** USER MESSAGE ****************
    with st.chat_message("user"):
        st.session_state["message_history"].append({
            "role": "user",
            "content": user_input
        })
        st.write(user_input)

    # ************** AI RESPONSE *****************
    with st.chat_message("assistant"):

        def stream_ai_only():
            for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content:
                        yield message_chunk.content


        ai_message = st.write_stream(stream_ai_only())
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })