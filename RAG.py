from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

def RAG_similarity(query_chunk: str, vectorstore: FAISS, llm: ChatOllama) -> dict:
    matches = vectorstore.similarity_search(query_chunk, k=3)

    if not matches:
        return {
            "query_chunk": query_chunk,
            "document_chunk": "", 
            "answer": "I don't know. No relevant documents were retrieved."
        }

    top_match = matches[0]

    prompt = f"""You are a helpful assistant who is good at analyzing source information and finding relationships between documents.

Use the following source documents to determine if there is any relationship between them.
If you don't know the answer, just say that you don't know.
Explain your reasoning clearly.
Use three sentences maximum and keep the answer concise.

First sentence: state whether a relationship exists or not.
Second sentence: explain why, citing shared concepts.
Third sentence (optional): mention uncertainty if any.

Document A:
{query_chunk}

Document B:
{top_match.page_content}
"""

    response = llm.invoke(prompt)

    return {
        "query_chunk": query_chunk,
        "document_chunk": top_match.page_content,
        "answer": response.content,
    }