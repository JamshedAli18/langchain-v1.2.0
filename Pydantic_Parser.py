from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# getting api keys from .env file
os.getenv("GROQ_API_KEY")

class Movie(BaseModel):
    title: str = Field(description="movie title")
    year: int = Field(description="release year")
    genre: str = Field(description="main genre")

parser = JsonOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie info as JSON.\n{format_instructions}"),

    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | ChatGroq(model_name="llama-3.1-8b-instant") | parser
result = chain.invoke({"text": "The Matrix came out in 1999 as a sci-fi film."})
print(result)
