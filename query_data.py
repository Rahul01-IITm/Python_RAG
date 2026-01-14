import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based strictly on the following context:

{context}

---

Question: {question}
Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    
    query_rag(args.query_text)

def query_rag(query_text: str):
    # 1. Load the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # 2. Search for the top 5 relevant PDF parts
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 3. Setup Kimi Model (using OpenAI-compatible interface)
    model = ChatOpenAI(
        model="moonshotai/kimi-k2-instruct-0905", 
        openai_api_key=os.environ.get("MOONSHOT_API_KEY"),
        openai_api_base=os.environ.get("MOONSHOT_BASE_URL")
    )

    # 4. Generate the response
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response = model.invoke(prompt)
    
    print("\n--- RESPONSE ---\n")
    print(response.content)
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print(f"\nSources used: {list(set(sources))}")

if __name__ == "__main__":
    main()