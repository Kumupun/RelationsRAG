from langchain_ollama import ChatOllama
from typing_extensions import Annotated, TypedDict

class EvalGrade(TypedDict):
    grounded: Annotated[bool, ..., "True if grounded in source"]
    relevant: Annotated[bool, ..., "True if answer is relevant to query"]
    retrieval_relevant: Annotated[bool, ..., "True if retrieved facts are relevant"]
    correct: Annotated[bool, ..., "True if answer is factually correct"]
    explanation: Annotated[str, ..., "Explain your reasoning for each criterion with a list 1. grounded, 2. relevant, 3. retrieval_relevant, 4. correct"]

def scores_result (results: list[EvalGrade]) -> dict[str, float, float]:
    total = len(results)
    if total == 0:
        return {metric: 0 for metric in ["grounded", "relevant", "retrieval_relevant", "correct"]}
    scores = {metric: (0, total, 0.0) for metric in ["grounded", "relevant", "retrieval_relevant", "correct"]}
    for _, eval in results:
        for metric in scores.keys():
            if eval[metric]:
                scores[metric] = (scores[metric][0] + 1, total, (scores[metric][0] + 1) / total )
            else:
                scores[metric] = (scores[metric][0], total, scores[metric][0] / total )
    return scores
eval_instructions = """
You are a strict RAG evaluator.

You will be given:

QUERY: chunk from Document A
FACTS: retrieved chunks from Document B
ANSWER: model relationship output

Evaluate:

1. grounded: 
(1) Ensure the ANSWER is grounded in the FACTS. 
(2) Ensure the ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

2. relevant:
(1) Ensure the ANSWER is concise and directly addresses the QUERY.
(2) Ensure the ANSWER completely addresses the query document without deviating into unrelated topics.

3. retrieval_relevant:
(1) If the FACTS contain ANY keywords or semantic meaning related to the QUERY, consider them relevant
(2) It is OK if the FACTS have SOME information that is unrelated to the QUERY as long as (1) is met

4. correctness:
(1) Grade the ANSWER based ONLY on its factual accuracy relative to the GROUND TRUTH.
(2) It is OK if the ANSWER is not directly grounded in the FACTS as long as it is factually correct.

Return booleans.
A value of True means that the ANSWER meets all of the criteria.
A value of False means that the ANSWER does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset.
"""
eval_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
).with_structured_output(
    EvalGrade,
    method="json_schema",
    strict=True
)

def evaluate(outputs: dict, truth: dict) -> dict:
    prompt = f"""
QUERY:
{outputs['query_chunk']}

FACTS:
{outputs['document_chunk']}

ANSWER:
{outputs['answer']}

GROUND TRUTH:
{truth['answer']}
"""

    grade = eval_llm.invoke([
        {"role": "system", "content": eval_instructions},
        {"role": "user", "content": prompt},
    ])

    return grade