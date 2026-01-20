import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")



# 2. Local Embeddings 
# Using 'all-MiniLM-L6-v2' - it's fast and runs on your CPU
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Your Data (The "Knowledge Base")
text_data = [
    "Kimi K2 is a 1-trillion parameter model by Moonshot AI.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Sentence Transformers convert text into math vectors."
]
documents = [Document(page_content=t) for t in text_data]

# 4. Create Vector Store (The search engine)
vector_db = FAISS.from_documents(documents, embeddings)

# 5. Define the LLM (Using Kimi K2 via OpenRouter)
llm = ChatOpenAI(
    model="moonshotai/kimi-k2:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)

# 6. The RAG Process
query = "Who made Kimi K2?"

# Step A: Retrieve relevant info from your data
docs = vector_db.similarity_search(query, k=1)
context = docs[0].page_content

# Step B: Generate answer using the context
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer based only on context:"
response = llm.invoke(prompt)

print(f"Response from Kimi K2: {response.content}")