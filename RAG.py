from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS


async def RAG_similarity(query_chunk: str, vectorstore: FAISS, llm: ChatOllama, threshold: float) -> dict:
    matches = vectorstore.similarity_search(query_chunk, k=3)

    threshold_match = [(doc, score)
        for doc, score in matches
        if score >= threshold]

    if not threshold_match:
        return {
            "query_chunk": query_chunk,
            "document_chunk": "", 
            "answer": "I don't know. No relevant documents were retrieved.",
            "similarity_score": None
        }

    top_match,score = threshold_match[0]

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

    response = await llm.ainvoke(prompt)

    return {
        "query_chunk": query_chunk,
        "document_chunk": top_match.page_content,
        "answer": response.content,
        "similarity_score": score
    }