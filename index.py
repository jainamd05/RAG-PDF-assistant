from pathlib import Path    # To read the PDF file into the python code
from langchain_community.document_loaders import PyPDFLoader # Load this file in python program
from langchain_text_splitters import RecursiveCharacterTextSplitter # Split docs into smaller chunks
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# * ---------------------------------------------------------------------------------------------------

pdf_path = Path(__file__).parent / "[FILE_NAME.TYPE]" # example : OS_module2.p

# * ---------------------------------------------------------------------------------------------------

# ! Load this file in python program
loader = PyPDFLoader(file_path = pdf_path)
docs = loader.load()    # Loads the PDF into page by page doc 
# print(docs[6])

print("PDF loaded Successfuly 👍")

# * ---------------------------------------------------------------------------------------------------

# ! Split docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)
chunks = text_splitter.split_documents(documents=docs)

print("Chunking of Document successfully done 👍")

# * ---------------------------------------------------------------------------------------------------

# ! Vector embeddings for chunks
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Importing of embedding model done 👍")

# * ---------------------------------------------------------------------------------------------------

# ! Vector storing in database
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url = "http://localhost:6333",
    collection_name = "DATABASE_NAME"
)
print("Indexing of documents done 👍")

# * ---------------------------------------------------------------------------------------------------