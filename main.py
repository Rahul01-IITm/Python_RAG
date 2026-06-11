import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


def main():
    # 1. Loading api key from .env
    load_dotenv() 

    # DEBUG: Check if key loaded
   
     


    # 2. Local Embeddings 
    print("loading embeddings...")   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3.Loading pdf documents
    pdf_path = "chess_rules.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    #3 chunking the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 4. Create Vector Store (The search engine)
    print("Building vector store...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("Database ready !")



    # 5. (Using QWEN via OpenRouter)
    llm = ChatGroq(
    model="qwen/qwen3-32b")

    while True:
        query = input("ask a question about your pdf")
        if query.lower() == "quit":
            break


        docs = vector_db.similarity_search(query, k=3)

   
        context = "\n\n".join([d.page_content for d in docs])

     # C. Create a prompt
        prompt = f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say you don't know.

Context:
{context}

Question: {query}
Answer:"""

        # D. Get Response
        print("\nThinking...")
        response = llm.invoke(prompt)
        print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    main()
