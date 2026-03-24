from langchain_core.documents import Document

doc = Document(
    page_content="This is the text content",
    metadata={"source": "file.pdf", "page": 1}
)

# print(doc)



# TextLoader — plain .txt files

from langchain_community.document_loaders import TextLoader

# loader = TextLoader("my_notes.txt")
# docs = loader.load()

# print(docs[0].page_content[:100])  # first 100 chars
# print(docs[0].metadata)  # {'source': 'my_notes.txt'}


# PyPDFLoader — PDF files

from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("Ai_engineer (2).pdf")
# docs = loader.load()

# print(len(docs))  # number of pages
# print(docs[0].page_content[:1000])  # first 100 chars of page 1
# print(docs[0].metadata)  # {'source': 'report.pdf', 'page': 0}



# WebBaseLoader — scrape a webpage

from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
# docs = loader.load()

# print(docs[0].page_content[:200])


#spreadSheets

from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("data.xlsx")
docs = loader.load()

print(docs[0].page_content)