from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag import rag_similarity_sync
from chunking import chunking
from tuner_class import ThresholdTuner
from eval_class import EvalGrade, evaluate

import json
import time

DOCUMENT_PATH = r"documents\doc_medical.txt"
QUERY_PATH = r"documents\medical_query.txt"
GROUND_TRUTH_PATH = r"documents\gt_medical.json"
RESULTS_PATH = r"documents\results_medical.json"
RAG_MODEL = "qwen3:0.6b"
JUDGE_MODEL = "llama3.1:8b"
EMBEDDING = "nomic-embed-text"
TOP_K_CHUNKS = 5

parag_split1 = chunking(DOCUMENT_PATH, chunk_size=300, chunk_overlap=50)
parag_split2 = chunking(QUERY_PATH, chunk_size=200, chunk_overlap=0)

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

rag_llm = ChatOllama(
    model= RAG_MODEL,
    temperature = 0.5)

eval_llm = ChatOllama(
    model= JUDGE_MODEL,
    temperature=0
).with_structured_output(
    EvalGrade,
    method="json_schema",
    strict=True
)

embeddings = OllamaEmbeddings(
    model= EMBEDDING
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
    rag_results = rag_similarity_sync(parag_split2, vectorstore, rag_llm, TOP_K_CHUNKS)
    print(f"Sample number {TOP_K_CHUNKS} retrieved for each query chunk.")

    t1 = time.perf_counter()
    
    results = [(chunk, [evaluate(chunk, truth, eval_llm) for truth in ground_truth]) for chunk in rag_results]
    
    json_output(results, RESULTS_PATH)
    print(f"Evaluation completed. Results saved to {RESULTS_PATH}")

    t2 = time.perf_counter()
    print(f"RAG processing completed in {t1 - t0:.2f} seconds.")
    print(f"Evaluation completed in {t2 - t1:.2f} seconds.")

    tuner = ThresholdTuner(ground_truth, results)
    best_threshold, best_score = tuner.tune()

    print(f"Best threshold: {best_threshold:.4f} with score: {best_score:.2%}")

    t3 = time.perf_counter()
    
    print(f"Tuning completed in {t3 - t2:.2f} seconds.")
    print(f"Total pipeline time: {t3 - t0:.2f} seconds.")
    return 

if __name__ == "__main__":
    main()

