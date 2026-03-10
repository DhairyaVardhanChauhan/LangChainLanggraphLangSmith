from typing import Annotated, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    name: Annotated[str, "Name"]
    amount: Annotated[int, "Amount"]
    transactionType: Annotated[str, "TransactionType"]
    date: Annotated[Optional[str], "Date"] = None
    cgpa: float = Field(gt=0, lt = 10,default=5,description="GPA")


test = Message(
    name="Dhairya",
    amount=5000,
    transactionType="TRANSFER",
    cgpa=9,
    # date=120 // will throw error
)

print(test)