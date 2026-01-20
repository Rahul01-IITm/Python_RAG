import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

def main():

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")



# 2. Local Embeddings 
# Using 'all-MiniLM-L6-v2' - it's fast and runs on your CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    if not os.path.exists("data.txt"):
        print("Error: data.txt not found")
        return

    loader = TextLoader("data.txt")
    documents = loader.load()

    #3 chunking the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 4. Create Vector Store (The search engine)
    print("Building vector store...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("Datbase ready !")



    # 5. (Using Kimi K2 via OpenRouter)
    llm = ChatOpenAI(
        model="moonshotai/kimi-k2:free",
        open_api_key=OPENROUTER_API_KEY,
        open_api_base="https://openrouter.ai/api/v1",
    )

    # 6. The RAG Process
    query = "Who made Kimi K2?"

# Step A: Retrieve relevant info from your data
    docs = vector_db.similarity_search(query, k=1)

    if not docs:
        print("No relevant documents found.")
        return
    context = docs[0].page_content

    # Step B: Generate answer using the context
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer based only on context:"
    response = llm.invoke(prompt)

    print(f"Response from Kimi K2:\n {response.content}")

if __name__ == "__main__":
    main()
