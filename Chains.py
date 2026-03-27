# Basic LCEL Chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# getting api keys from .env file
os.getenv("GROQ_API_KEY")

# Build chain using | operator
chain = (
    ChatPromptTemplate.from_template("Summarize this in two to three words: {text}")
    | ChatGroq(model_name="llama-3.1-8b-instant")
    | StrOutputParser()
)

result = chain.invoke({"text": "The Earth orbits the Sun once every 365.25 days..."})
# print(result)

# Sequential Chain (chain of chains)

translate_chain = (
    ChatPromptTemplate.from_template("Translate {text} to {language}")
    | ChatGroq(model_name="llama-3.1-8b-instant")
    | StrOutputParser()
)


summarize_chain = (
    ChatPromptTemplate.from_template("Summarize this in two to three words: {text}")
    | ChatGroq(model_name="llama-3.1-8b-instant")
    | StrOutputParser()
)

sequential_chain = (
    translate_chain
    | summarize_chain
)

result = sequential_chain.invoke({"text": "The Earth orbits the Sun once every 365.25 days...", "language": "French"})
# print(result)


# Parallel Chain (chain of chains)

from langchain_core.runnables import RunnableParallel

# Two chains that run in parallel on the same input
parallel = RunnableParallel(
    summary=ChatPromptTemplate.from_template("Write summary of this: {topic}") | ChatGroq(model_name="llama-3.1-8b-instant") | StrOutputParser(),
    keywords=ChatPromptTemplate.from_template("List keywords from: {topic}") | ChatGroq(model_name="llama-3.1-8b-instant") | StrOutputParser()
)

result = parallel.invoke({"topic": "Machine learning uses data to train models."})
print(result["summary"])   # "ML trains models on data."
print(result["keywords"])  # "machine learning, data, models, training"
