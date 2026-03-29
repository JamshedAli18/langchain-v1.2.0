# FAISS — fast, local, in-memory

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

# embedder = OpenAIEmbeddings()

# # Create from documents
# docs = [...]  # your split documents
# vectorstore = FAISS.from_documents(docs, embedder)

# # Similarity search
# results = vectorstore.similarity_search("What is Python?", k=3)
# for doc in results:
#     print(doc.page_content[:100])

# # Save and load
# vectorstore.save_local("faiss_index")
# vectorstore = FAISS.load_local("faiss_index", embedder,
#     allow_dangerous_deserialization=True)




# Chroma — persistent, easy setup
# pip install chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

#loading the document
loader = PyPDFLoader("Ai_engineer (2).pdf")
docs = loader.load()

# print(len(docs))  # number of pages
# print(docs[0].page_content[:1000])  # first 100 chars of page 1
# print(docs[0].metadata)  # {'source': 'report.pdf', 'page': 0}

#splitting the document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max characters per chunk
    chunk_overlap=200,    # overlap between chunks (preserves context)
)

chunks = splitter.split_documents(docs)


#loading the embeddings
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create and persist
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory="./chroma_db"   # saved to disk
)

# Later: reload it
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedder
# )

# Search
results = vectorstore.similarity_search("Education", k=1)

for doc in results:
    print(doc.page_content[:100])
