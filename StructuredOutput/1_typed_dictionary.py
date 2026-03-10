from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI()
class Message(TypedDict):
    bankCode: Annotated[str, "Bank Code"]
    amount: Annotated[int, "Amount"]
    transactionType: Annotated[str,"If transfer then debit if received them credit"]
    date: Annotated[Optional[str],"If present in the message in dd/mm/yyyy format(write numerical date)"]

# poem.txt:Message = {
#     "bankCode": "Bank Code",
#     "amount": "Amount",
# }
llm_structured = llm.with_structured_output(Message)
res1 = llm.invoke("Transfer 5000 rupees using bank code HDFC123")
res2 = llm_structured.invoke("Transfer 5000 rupees using bank code HDFC123 on 23 jan 2012")

print(f"Non structured {res1}")
print(f"Structured {res2}")

