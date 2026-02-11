
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import ChatOllama, OllamaEmbeddings

from RAG import RAG_similarity
from Chunking import chunking
from  Score_class import ThresholdTuner
from Eval_class import evaluate, scores_result


import json
import time

doc1_path = r"Documents\Doc1.txt"
doc2_path = r"Documents\Doc2.txt"

parag_split1 = chunking(doc1_path, chunk_size=300, chunk_overlap=50)
parag_split2 = chunking(doc2_path, chunk_size=200, chunk_overlap=0)

with open(r"Documents\ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

llm = ChatOllama(
    model="qwen3:8b",
    temperature = 0.5)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = FAISS.from_documents(
    parag_split1,
    embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

def json_output(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

async def main():
    t0 = time.perf_counter()
    tuner = ThresholdTuner(vectorstore, ground_truth, llm, parag_split2, evaluate, scores_result)
    best_threshold, best_score = await tuner.tune()

    print(f"Best threshold: {best_threshold:.4f} with score: {best_score:.2%}")

    t1 = time.perf_counter()
    rag_process = [RAG_similarity(chunk.page_content, vectorstore, llm, best_threshold) for chunk in parag_split2]
    rag_results = await asyncio.gather(*rag_process)
    t2 = time.perf_counter()
    
    results = []
    for chunk, truth in zip(rag_results, ground_truth):
        eval =  evaluate(chunk, truth)
        results.append((chunk, eval))

    json_output(results, "results.json")
    print("Evaluation completed. Results saved to results.json")

    t3 = time.perf_counter()

    print(f"Tuning completed in {t1 - t0:.2f} seconds.")
    print(f"RAG processing completed in {t2 - t1:.2f} seconds.")
    print(f"Evaluation completed in {t3 - t2:.2f} seconds.")
    print(f"Total pipeline time: {t3 - t0:.2f} seconds.")
    return 

if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()

    


