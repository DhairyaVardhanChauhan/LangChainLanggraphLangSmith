from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from rich.pretty import pprint

load_dotenv()

search_tool = DuckDuckGoSearchRun()
llm_client = ChatAnthropic(model_name="claude-sonnet-4-6")

agent = create_agent(llm_client, tools=[search_tool])

prompt = "What is the capital of France and what is its population?"

response = agent.invoke({
    "messages": [HumanMessage(content=prompt)]
})

pprint(response)