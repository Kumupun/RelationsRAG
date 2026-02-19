from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

def RAG_similarity_sync(parag2,vectorstore , llm, num):
    results = []
    for chunk in parag2:
        matches = vectorstore.similarity_search_with_relevance_scores(chunk.page_content, k=num)
        for top_match,score in matches:

            prompt = f"""You are a helpful assistant who is good at analyzing source information and finding relationships between documents.

Use the following source documents to determine if there is any relationship between them.
If you don't know the answer, just say that you don't know.
Explain your reasoning clearly.
Use three sentences maximum and keep the answer concise.

First sentence: state whether a relationship exists or not.
Second sentence: explain why, citing shared concepts.
Third sentence (optional): mention uncertainty if any.

Document A:
{chunk.page_content}

Document B:
{top_match.page_content}
"""

            response = llm.invoke(prompt)
            results.append({
                "query_chunk": chunk.page_content,
                "document_chunk": top_match.page_content,
                "answer": response.content,
                "score": float(score)
            })
    return results