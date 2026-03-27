from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# getting api keys from .env file
os.getenv("GROQ_API_KEY")

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([

    ("system", "Give the opposite of the word."),
    few_shot_prompt,
    ("human", "{word}")
])

# chain

chain = final_prompt | ChatGroq(model_name="llama-3.1-8b-instant")

result = chain.invoke({"word": "Male"})

print("Result: ",result.content)
