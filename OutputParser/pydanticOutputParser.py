from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatAnthropic(
    model_name="claude-sonnet-4-6"
)


class Person(BaseModel):
    name: str = Field(description= "Name of the person")
    age: int = Field(description="Age of the person")



class RichestPeople(BaseModel):
    people: List[Person]
parser = PydanticOutputParser(pydantic_object=RichestPeople)
template1 = PromptTemplate(
    template="Top 3 richest man in india \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
print(template1)
chain = template1 | model | parser
# print(chain.get_graph().print_ascii())
result = chain.invoke({})
print(result)



