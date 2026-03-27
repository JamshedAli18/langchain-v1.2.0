from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# getting api keys from .env file
os.getenv("GROQ_API_KEY")

messages = [
    SystemMessage(content="You are a coding tutor."),
    HumanMessage(content="What is a list?"),
    AIMessage(content="A list is an ordered collection..."),
    HumanMessage(content="Can you give an example?")
]

llm = ChatGroq(model_name="llama-3.1-8b-instant")
result = llm.invoke(messages)
print("Result: ",result.content)
