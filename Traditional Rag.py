from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ── 1. LOAD ──────────────────────────────────────────
loader = PyPDFLoader("Ai_engineer (2).pdf")
docs = loader.load()

# ── 2. SPLIT ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ── 3. EMBED + STORE ─────────────────────────────────
vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
))
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── 4. PROMPT ────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If you don't know, say "I don't know."

Context:
{context}

Question: {question}
""")

# ── 5. CHAIN ─────────────────────────────────────────
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 6. QUERY ─────────────────────────────────────────
answer = rag_chain.invoke("from which university jamshed have done his bachelors")
print(answer)
