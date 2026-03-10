from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = ChatAnthropic(model_name="claude-sonnet-4-6")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Return exactly {num} random numbers separated by commas only.",
    input_variables=["num"]
)

prompt2 = PromptTemplate(
    template="Return exactly {num} random words separated by commas only.",
    input_variables=["num"]
)

prompt3 = PromptTemplate(
    template="""You are given:

Numbers: {numbers}
Strings: {strings}

Merge them alternately (number, string, number, string...) 
Return only the merged comma-separated result with {text} at the end!.
""",
    input_variables=["numbers","strings","text"]
)

parallel_chain = RunnableParallel({
    "numbers": prompt1 | model | parser,
    "strings": prompt2 | model | parser,
    "text": RunnableLambda(lambda _: "%%%%")
})

chain = prompt3 | model | parser

final_chain = parallel_chain | chain

result = final_chain.invoke({"num": 5})

print(final_chain.get_graph().print_ascii())
print(result)