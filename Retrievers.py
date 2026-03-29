from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder
)

# Convert vectorstore to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr"
    search_kwargs={"k": 4}     # return top 4 chunks
)

# Use it
results = retriever.invoke("What is the education of Jamshed Ali?")
print(results)
# returns: List[Document]
