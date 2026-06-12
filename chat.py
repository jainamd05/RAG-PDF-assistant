from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(
    api_key = api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ! Vector Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "OS_notes",
    embedding = embedding_model
)


# Take user input
user_query = input("👉 : ")

# ! Retrieving chunks from the vector db (Simi similarity search)
search_results = vector_db.similarity_search(query = user_query)

context = "\n\n\n".join([f"Page Content : {result.page_content}\nPage Number : {result.metadata['page_label']}\nFile Location : {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
    You are an helpful AI assistaant that answers user query based on available context, 
    retrieved from the PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the 
    user to open the right page number to know more.

    Context: {context}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user", "content": user_query}
    ]
)

print(f"🤖 : {response.choices[0].message.content}")