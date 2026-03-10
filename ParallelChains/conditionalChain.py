from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch, RunnablePassthrough
from pydantic import BaseModel

load_dotenv()

class FeedBack(BaseModel):
    type: Literal["positive", "negative"]

model = ChatAnthropic(model_name="claude-sonnet-4-6")
parser = PydanticOutputParser(pydantic_object=FeedBack)
strParser = StrOutputParser()
template = PromptTemplate(template="Classify the feedback(positive/negative) \n {feedback} \n {format}",input_variables=["feedback"],partial_variables={"format":parser.get_format_instructions()})
chain = template|model|parser

good_template = PromptTemplate(template= "Write an appropriate response for the positive feedBack: {feedback}",input_variables=["feedback"])
bad_template = PromptTemplate(template= "Write an appropriate response for the negative feedBack: {feedback}",input_variables=["feedback"])


final_chain = (
    RunnableBranch(
        (
            lambda x: x.type == "positive",
            good_template | model | strParser,
        ),
        (
            lambda x: x.type == "negative",
            bad_template | model | strParser,
        ),
        RunnableLambda(lambda x: "Could not classify the feedback properly."),
    )
)

resultChain = chain | final_chain
print(resultChain.invoke({"feedback": "My phone screen shattered in the pocket!?"}))