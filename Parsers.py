from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# getting api keys from .env file
os.getenv("GROQ_API_KEY")

chain = (
    ChatPromptTemplate.from_template("Tell me a fact about {topic}")
    | ChatGroq(model_name="llama-3.1-8b-instant")
    | StrOutputParser()  # extracts .content as plain string
)

result = chain.invoke({"topic": "space"})
print(result)       # "The Sun is 109 times wider than Earth."
print(type(result)) # str


