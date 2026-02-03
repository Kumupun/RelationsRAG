from langchain_ollama import ChatOllama
from typing_extensions import Annotated, TypedDict






class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a competent secretary. You will be given two documents with determined relationship and a ground truth relationship. Here is the grade criteria to follow:
(1) Grade the given relationship between the documents based ONLY on their factual accuracy relative to the ground truth relationship.
(2) It is OK if the relationship answer contains more information than the ground truth, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the relationship answer meets all of the criteria.
A correctness value of False means that the relationship answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

grader_llm = ChatOllama(model="qwen3:8b", temperature = 0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness(outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUERY: {outputs['query_chunk']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
RELATIONSHIP: {outputs['answer']}"""
    
    grade = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on if the answer is relevant to the query document"
    ]   
relevance_instructions = """You are a competent secretary. You will be given two documents with determined relationship. Here is the grade criteria to follow:
(1) Ensure the relationship is concise and relevant to the query document.
(2) Ensure the relationship completely addresses the query document without deviating into unrelated topics.

Relevance:
A relevance value of True means that the relationship meets all of the criteria.
A relevance value of False means that the relationship does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

relevance_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

def relevance(outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUERY: {outputs['query_chunk']}\nRELATIONSHIP: {outputs['answer']}"
    grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "True if the relationship is grounded in the source document, False otherwise"]

grounded_instructions = """You are a competent secretary. You will be given two documents with determined relationship. Here is the grade criteria to follow:
(1) Ensure the relationship is grounded in the source document. (2) Ensure the relationship does not contain "hallucinated" information outside the scope of the source document.

Grounded:
A grounded value of True means that the relationship meets all of the criteria.
A grounded value of False means that the relationship does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

grounded_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

def groundedness(outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    answer = f"FACTS: {outputs['document_chunk']}\nRELATIONSHIP: {outputs['answer']}"
    grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved source documents are relevant to the query document, False otherwise",
    ]

retrieval_relevance_instructions = """You are a competent secretary. You will be given two documents with determined relationship. You goal is to identify facts from source documents that are completely unrelated to the query.
Here is the grade criteria to follow:
(1) If the facts contain ANY keywords or semantic meaning related to the query, consider them relevant
(2) It is OK if the facts have SOME information that is unrelated to the query as long as (1) is met

Relevance:
A relevance value of True means that the facts contain ANY keywords or semantic meaning related to the query and are therefore relevant.
A relevance value of False means that the facts are completely unrelated to the query.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
).with_structured_output(
    RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(outputs: dict) -> bool:
    """An evaluator for document relevance"""
    answer = f"FACTS: {outputs['document_chunk']}\nQUERY: {outputs['query_chunk']}"
    grade = retrieval_relevance_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]