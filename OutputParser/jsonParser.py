from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatAnthropic(
    model_name="claude-sonnet-4-6"
)

parser = JsonOutputParser()
template1 = PromptTemplate(
    template="Top 3 richest man in india \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
print(template1)
chain = template1 | model | parser
result = chain.invoke({})
print(result)


# aise hi ek structured output parser aur pydantic output parser hota hai....



