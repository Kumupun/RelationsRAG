
def rag_similarity_sync(document: list[Document], vectorstore: FAISS , llm: ChatOllama, num: int) -> dict:
    results = []
    for chunk in document:
        matches = vectorstore.similarity_search_with_relevance_scores(chunk.page_content, k=num)
        for top_match,score in matches:

            prompt = f""" 
You are an analytical assistant tasked with identifying explicit semantic relationships between two documents based only on the provided source text. 
Determine whether a relationship exists between the documents. 
Output rules (strict): Use at most three sentences. 
Do not add external knowledge or assumptions. 
Do not speculate beyond the given text. 
Response structure: 
Sentence 1: Clearly state whether a relationship exists (e.g., “A relationship exists” or “No relationship exists”). 
Sentence 2: Justify the decision by referencing specific shared concepts, actions, or claims found in both documents. 
Sentence 3 (optional): Briefly state uncertainty only if the evidence is weak or implicit. 
If the documents discuss unrelated topics or lack overlapping concepts, strictly respond "I don't know. No relevant documents were retrieved." without any additional commentary.

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
