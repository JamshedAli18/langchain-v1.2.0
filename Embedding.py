from langchain_community.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # small & fast
)

# vector = embedder.embed_query("Hello world")
# print(len(vector))  # 384 dimensions

# Embed multiple documents at once
vectors = embedder.embed_documents([
    "Python is a programming language.",
    "Dogs are loyal animals.",
    "Paris is the capital of France."
])
print(len(vectors[1]))  # 3 vectors, each 384 dimensions

