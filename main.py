from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import ChatOllama, OllamaEmbeddings

from RAG import RAG_similarity_sync
from Chunking import chunking
from Tuner_class import ThresholdTuner
from Eval_class import evaluate

import json
import time

DOCUMENT_PATH = r"Documents\Doc_medical.txt"
QUERY_PATH = r"Documents\Medical_query.txt"
GROUND_TRUTH_PATH = r"Documents\GT_medical.json"
RESULTS_PATH = r"Documents\results_medical.json"

parag_split1 = chunking(DOCUMENT_PATH, chunk_size=300, chunk_overlap=50)
parag_split2 = chunking(QUERY_PATH, chunk_size=200, chunk_overlap=0)

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

llm = ChatOllama(
    model="phi",
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

def main():
    t0 = time.perf_counter()
    k=5
    rag_results = RAG_similarity_sync(parag_split2, vectorstore, llm, k)
    print(f"Sample number {k} retrieved for each query chunk.")

    t1 = time.perf_counter()

    results = [(chunk, [evaluate(chunk, truth) for truth in ground_truth]) for chunk in rag_results]
    
    json_output(results, RESULTS_PATH)
    print(f"Evaluation completed. Results saved to {RESULTS_PATH}")

    t2 = time.perf_counter()
    print(f"RAG processing completed in {t1 - t0:.2f} seconds.")
    print(f"Evaluation completed in {t2 - t1:.2f} seconds.")

    tuner = ThresholdTuner(ground_truth, rag_results, evaluate)
    best_threshold, best_score = tuner.tune()

    print(f"Best threshold: {best_threshold:.4f} with score: {best_score:.2%}")

    t3 = time.perf_counter()
    
    print(f"Tuning completed in {t3 - t2:.2f} seconds.")
    print(f"Total pipeline time: {t3 - t0:.2f} seconds.")
    return 

if __name__ == "__main__":
    main()

