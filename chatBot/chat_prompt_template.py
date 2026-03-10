import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

model = ChatOpenAI()

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "Dusra"
})

print(prompt)

result = model.invoke(prompt)

print("==============")
print(result.content)