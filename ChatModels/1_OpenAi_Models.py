from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-4")
ers = model.invoke("What is the capital of India")
print(ers.content)