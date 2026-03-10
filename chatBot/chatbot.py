from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
chat_history = [
    SystemMessage(content="You are a helpful ai assistant"),
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI: ",result.content)

print(chat_history)