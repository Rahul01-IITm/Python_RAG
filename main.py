import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


def main():
    # 1. Loading api key from .env
    load_dotenv() 
    my_key = os.getenv("OPENROUTER_API_KEY")

    # DEBUG: Check if key loaded
    if not my_key:
        print(" Missing Api key!")
        return
     


    # 2. Local Embeddings 
    print("loading embeddings...")   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3.Loading pdf documents
    pdf_path = "chess_rules.pdf"
    if not os.path.exists("pdf_path"):
        print("Error: {pdf_path} not found")
        return

    loader = PyPDFLoader("pdf_path")
    documents = loader.load()

    #3 chunking the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 4. Create Vector Store (The search engine)
    print("Building vector store...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("Datbase ready !")



    # 5. (Using Kimi K2 via OpenRouter)
    llm = ChatOpenAI(
    model="moonshotai/kimi-k2:free",
    openai_api_key=my_key,
    base_url="https://openrouter.ai/api/v1",)

    while True:
        query = input("ask a question about your pdf")
        if query.lower() == "quit":
            break


        docs = vector_db.similarity_search(query, k=1)

   
        context = docs[0].page_content

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
