from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)
model = ChatHuggingFace(llm= llm)
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=[
        'topic'
    ]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text {text}",
    input_variables=[
        'text'
    ]
)
print("Here")
prompt1 = template1.invoke({"topic":"Tell me about epstine files"})
print("Here")
output = model.invoke(prompt1)
print("Here")
prompt2 = template2.invoke({"text":output.content})
print("Here")
finalOutput = model.invoke(prompt2)
print("Here")
print(finalOutput)

