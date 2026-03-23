from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
 
# getting api keys from .env file
os.getenv("GROQ_API_KEY")

# Calling Groq models with langchain
groq = ChatGroq(model_name="openai/gpt-oss-120b")

template = PromptTemplate(
    input_variables=["product", "language"],
    template="Write a tagline for {product} in {language}."
)

prompt = template.format(product="coffee", language="English")
print("Prompt: ",prompt)


# Creating a chain
chain = template | groq

# Invoking the chain
result = chain.invoke({"product": "coffee", "language": "English"})

print("Result: ",result.content)
