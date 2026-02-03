
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_ollama import ChatOllama, OllamaEmbeddings

from RAG import RAG_similarity
from Chunking import chunking
# from Evaluation import relevance, groundedness, retrieval_relevance
from Eval_class import evaluate

doc1_path = "Doc1.txt"
doc2_path = "Doc2.txt"

parag_split1 = chunking(doc1_path, chunk_size=300, chunk_overlap=50)
parag_split2 = chunking(doc2_path, chunk_size=150, chunk_overlap=0)

import json

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

results = []
for chunk,truth in zip(parag_split2, ground_truth):
    output = RAG_similarity(chunk.page_content, vectorstore, llm)
    eval_scores = evaluate(output, truth)
    results.append((output, eval_scores))

for output, scores in results:
    print("=" * 80)
    print("QUERY:")
    print(output["query_chunk"])
    print("\nDOCUMENT:")
    print(output["document_chunk"])
    print("\nANSWER:")
    print(output["answer"])
    print("\nEVALUATION SCORES:")
    for k, v in scores.items():
        print(f"  {k}: {v}")