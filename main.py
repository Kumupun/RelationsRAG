
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import ChatOllama, OllamaEmbeddings

from RAG import RAG_similarity
from Chunking import chunking
# from Evaluation import relevance, groundedness, retrieval_relevance
from Eval_class import evaluate

import json
import time

doc1_path = "Doc1.txt"
doc2_path = "Doc2.txt"

parag_split1 = chunking(doc1_path, chunk_size=300, chunk_overlap=50)
parag_split2 = chunking(doc2_path, chunk_size=200, chunk_overlap=0)

with open("ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

llm = ChatOllama(
    model="qwen3:8b",
    temperature = 1)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = FAISS.from_documents(
    parag_split1,
    embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

# for chunk in parag_split2:
#     output = RAG_similarity(chunk.page_content, vectorstore, llm)

#     scores = {
#         "grounded": groundedness(output),
#         "relevant": relevance(output),
#         "retrieval_relevant": retrieval_relevance(output)
#     }

#     results.append((output, scores))
def json_output(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

async def main():
    results = []
    t0 = time.perf_counter()
    rag_process = [RAG_similarity(chunk.page_content, vectorstore, llm) for chunk in parag_split2]
    rag_results = await asyncio.gather(*rag_process)
    t1 = time.perf_counter()
    print(f"RAG processing completed in {t1 - t0:.2f} seconds.")

    for chunk, truth in zip(rag_results, ground_truth):
        eval =  evaluate(chunk, truth)
        results.append((chunk, eval))
    json_output(results, "results.json")
    
    t2 = time.perf_counter()
    print(f"Evaluation completed in {t2 - t1:.2f} seconds.")
    print(f"Total pipeline time: {t2 - t0:.2f} seconds.")
    print("Evaluation completed. Results saved to results.json")
    return 

if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()

    


