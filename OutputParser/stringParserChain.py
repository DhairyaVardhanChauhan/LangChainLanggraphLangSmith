from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatAnthropic(
    model_name="claude-sonnet-4-6"
)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=[
        'topic'
    ]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text {text} ",
    input_variables=[
        'text'
    ]
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic":"What is epstine files and how is PM modi related to it?"})
print(result)



