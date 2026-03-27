from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=3,
    api_key=os.getenv("GROQ_API_KEY")
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

compressor = LLMChainExtractor.from_llm(
    llm
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = retriever.invoke("")

print(results)

