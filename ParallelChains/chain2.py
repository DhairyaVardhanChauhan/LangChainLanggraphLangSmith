
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()
model = ChatAnthropic(
    model_name="claude-sonnet-4-6"
)



class Op(BaseModel):
    summary: str = Field(description="Summary of the joke")
    length: int = Field(description="Number of words in the joke")

template1 = PromptTemplate(template="Give me a joke on topic {topic}",input_variables=["topic"])
template2 = PromptTemplate(template="Explain me this joke {joke}",input_variables=["joke"])
parser = StrOutputParser()
ans_parser = PydanticOutputParser(pydantic_object=Op)

chain = template1 | model | parser

parallel_chain = RunnableParallel(
    {
        "summary":  RunnableLambda(lambda x: {"joke": x}) | template2 | model | parser,
        "length": RunnableLambda(lambda x:len(x.split(' ')))
    }
)

final_chain = chain | parallel_chain |RunnableLambda(lambda x: Op(**x))
res = final_chain.invoke({"topic":"AI"})
print(res.summary)
print(res.length)