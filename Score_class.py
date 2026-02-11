import numpy as np
import asyncio
from RAG import RAG_similarity

class ThresholdTuner:
    def __init__(self, vectorstore, ground_truth, higher_is_better=True):
        self.vectorstore = vectorstore
        self.ground_truth = ground_truth
        self.higher_is_better = higher_is_better

    async def evaluate_threshold(self, threshold):

        rag_tasks = [
            RAG_similarity(
                chunk.page_content,
                self.vectorstore,
                self.llm,
                threshold
            )
            for chunk in self.query_chunks
        ]

        rag_results = await asyncio.gather(*rag_tasks)

        results = []

        for output, truth in zip(rag_results, self.ground_truth):
            eval_result = self.evaluate(output, truth)
            results.append((output, eval_result))

        scores = self.scores_result(results)

        weighted_score = sum(
            scores[m][2] * scores[m][1] for m in scores.keys()
        ) / sum(scores[m][1] for m in scores.keys())

        return weighted_score

    async def tune(self):
        raw_scores = []

        for chunk in self.query_chunks:
            matches = self.vectorstore.similarity_search_with_score(
                chunk.page_content,
                k=1
            )
            if matches:
                _, score = matches[0]
                raw_scores.append(score)

        min_s = min(raw_scores)
        max_s = max(raw_scores)

        thresholds = np.linspace(min_s, max_s, 15)

        best_threshold = None
        best_score = 0

        for t in thresholds:
            print(f"Testing threshold: {t:.4f}")

            score = await self.evaluate_threshold(t)

            print(f"Weighted score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_threshold = t

        return best_threshold, best_score