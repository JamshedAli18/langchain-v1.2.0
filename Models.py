from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


# getting api keys from .env file
os.getenv("GROQ_API_KEY")
os.getenv("NVIDIA_API_KEY")
os.getenv("OPENAI_API_KEY")
os.getenv("ANTHROPIC_API_KEY")
os.getenv("GOOGLE_API_KEY")



# Calling Groq models with langchain
groq = ChatGroq(model_name="openai/gpt-oss-120b")

result = groq.invoke("Tell me something about Langchain")

print("Groq: ",result.content)



# Calling NVIDIA models with langchain
nvidia = ChatNVIDIA(model_name="meta-llama/llama-3.1-8b-instant")

result = nvidia.invoke("Tell me something about Langchain")

print("NVIDIA: ",result.content)



# Calling OpenAI models with langchain
openai = ChatOpenAI(model_name="gpt-4o")

result = openai.invoke("Tell me something about Langchain")

print("OpenAI: ",result.content)



# Calling Anthropic models with langchain
anthropic = ChatAnthropic(model_name="claude-3-haiku-20240307")

result = anthropic.invoke("Tell me something about Langchain")

print("Anthropic: ",result.content)



# Calling Google models with langchain
google = ChatGoogleGenerativeAI(model_name="gemini-1.5-flash")

result = google.invoke("Tell me something about Langchain")

print("Google: ",result.content)
