from dotenv import load_dotenv

from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature = 0)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    #model = "text-embedding-004"
)
    
doc1_path = "Doc1.txt"
doc2_path = "Doc2.txt"

def loader(file_path: str) :
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

try:
    doc1_text = loader(doc1_path)
    doc2_text = loader(doc2_path)
    print("Documents have been loaded successfully.")
except Exception as e:
    print(f"Error loading documents: {e}")
    raise   

doc1 = [Document(page_content=doc1_text, metadata={"source": "doc1"})]
doc2 = [Document(page_content=doc2_text, metadata={"source": "doc2"})]

text_splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?"]
)

text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", "!", "?"]
)
parag_split1 = text_splitter1.split_documents(doc1)
parag_split2 = text_splitter2.split_documents(doc2)

vectorstore = FAISS.from_documents(
    parag_split1,
    embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

# def analyse_similarity():  
#     relationships = []
#     for chunk in parag_split2:
#         matches = vectorstore.similarity_search(
#         chunk.page_content, k=3)
#         for match in matches:
#             relationships.append({
#                 "query_chunk": chunk.page_content,
#                 "document_chunk": match.page_content
#             })
#     return relationships

def RAG_similarity(query_chunk: str) -> dict:
    matches = vectorstore.similarity_search(query_chunk, k=3)

    if not matches:
        return {
            "query_chunk": query_chunk,
            "document_chunk": "",
            "answer": "I don't know. No relevant documents were retrieved."
        }

    top_match = matches[0]

    prompt = f"""You are a helpful assistant who is good at analyzing source information and finding relationships between documents.

Use the following source documents to determine if there is any relationship between them.
If you don't know the answer, just say that you don't know.
Explain your reasoning clearly.
Use three sentences maximum and keep the answer concise.

First sentence: state whether a relationship exists or not.
Second sentence: explain why, citing shared concepts.
Third sentence (optional): mention uncertainty if any.

Document A:
{query_chunk}

Document B:
{top_match.page_content}
"""

    response = llm.invoke(prompt)

    return {
        "query_chunk": query_chunk,
        "document_chunk": top_match.page_content,
        "answer": response.content,
    }


# class CorrectnessGrade(TypedDict):
#     explanation: Annotated[str, ..., "Explain your reasoning for the score"]
#     correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# correctness_instructions = """You are a competent secretary. You will be given two documents with determined relationship and a ground truth relationship. Here is the grade criteria to follow:
# (1) Grade the given relationship between the documents based ONLY on their factual accuracy relative to the ground truth relationship.
# (2) It is OK if the relationship answer contains more information than the ground truth, as long as it is factually accurate relative to the ground truth answer.

# Correctness:
# A correctness value of True means that the relationship answer meets all of the criteria.
# A correctness value of False means that the relationship answer does not meet all of the criteria.

# Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# grader_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0).with_structured_output(
#     CorrectnessGrade, method="json_schema", strict=True
# )

# def correctness(outputs: dict, reference_outputs: dict) -> bool:
#     """An evaluator for RAG answer accuracy"""
#     answers = f"""\
# QUERY: {outputs['query_chunk']}
# GROUND TRUTH ANSWER: {reference_outputs['answer']}
# RELATIONSHIP: {outputs['answer']}"""
    
#     grade = grader_llm.invoke([
#             {"role": "system", "content": correctness_instructions},
#             {"role": "user", "content": answers},
#         ]
#     )
#     return grade["correct"]

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

relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0).with_structured_output(
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
        bool, ..., "Provide the score on if the answer hallucinates from the source documents"
    ]

grounded_instructions = """You are a competent secretary. You will be given two documents with determined relationship. Here is the grade criteria to follow:
(1) Ensure the relationship is grounded in the source document. (2) Ensure the relationship does not contain "hallucinated" information outside the scope of the source document.

Grounded:
A grounded value of True means that the relationship meets all of the criteria.
A grounded value of False means that the relationship does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

grounded_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(
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

retrieval_relevance_instructions = """You are a competent secretary. You will be given two documents with determined relationship. Here is the grade criteria to follow:
(1) You goal is to identify facts from source documents that are completely unrelated to the query.
(2) If the facts contain ANY keywords or semantic meaning related to the query, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the query as long as (2) is met

Relevance:
A relevance value of True means that the facts contain ANY keywords or semantic meaning related to the query and are therefore relevant.
A relevance value of False means that the facts are completely unrelated to the query.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(
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

results = []
for chunk in parag_split2:
    output = RAG_similarity(chunk.page_content)

    scores = {
        "grounded": groundedness(output),
        "relevant": relevance(output),
        "retrieval_relevant": retrieval_relevance(output)
    }

    results.append((output, scores))

for output, scores in results:
    print("=" * 80)
    print("QUERY:")
    print(output["query_chunk"])
    print("\nANSWER:")
    print(output["answer"])
    print("\nSCORES:")
    for k, v in scores.items():
        print(f"  {k}: {v}")