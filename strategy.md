# RelationsRAG Strategy

## Overview

This project implements a Retrieval-Augmented Generation to automatically detect and evaluate relationships between two documents.

Document A (`Doc1.txt`) is indexed in a vector database and acts as the knowledge base.  
Document B (`Doc2.txt`) is treated as the query document and is processed chunk-by-chunk.

For each chunk in Document B, the system retrieves relevant chunks from Document A, asks an LLM to determine whether a relationship exists, and then evaluates the generated relationship using LLM-as-judge metrics.

The goal is to identify concise, grounded relationships such as agreement, contradiction, or semantic overlap between the two documents.

## High-Level Architecture

1. Load documents
2. Split documents into chunks
3. Embed Document A and store in FAISS
4. For each chunk in Document B:
   - Retrieve top-k similar chunks from Document A
   - Ask an LLM to infer a relationship
   - Evaluate the result using multiple LLM-based judges


## Document Processing

### Loading

- `Doc1.txt` - treated as source document (Document A)
- `Doc2.txt` - treated as query document (Document B)

Both are loaded as LangChain `Document` objects with metadata indicating their origin.


### Chunking Strategy

Different chunking strategies are used for each document:

#### Document A (knowledge base)
- Chunk size: 300
- Overlap: 50

This preserves context continuity for retrieval.

#### Document B (query document)
- Chunk size: 200
- Overlap: 0

Each chunk is treated as an independent query unit.


## Embeddings & Vector Store

- Embeddings: `models/gemini-embedding-001`
- Vector store: FAISS
- Distance metric: Cosine similarity

Only Document A chunks are embedded and indexed.

Document B chunks are used purely as queries.


## Retrieval

For each chunk in Document B:

- Perform similarity search against FAISS
- Retrieve top 3 matches
- Select the top match as Document B’s counterpart

If no matches are found, the system returns “I don’t know.”


## Relationship Generation (RAG)

For every query chunk, the system prompts Gemini with:

- Document B chunk (query)
- Retrieved Document A chunk (source)

The LLM is instructed to:

- State whether a relationship exists
- Explain shared concepts
- Keep output within three sentences

This produces a concise relationship statement such as:

- Relationship exists due to shared themes
- No relationship found
- Partial overlap with uncertainty


## Evaluation: LLM-as-Judge

After generating a relationship, three independent evaluators are applied.

Each evaluator uses Gemini 2.5 Flash with structured JSON output.


### 1. Relevance

Checks whether the generated relationship:

- Is concise
- Directly addresses the query chunk
- Avoids unrelated topics

Output: `True / False`


### 2. Groundedness

Checks whether the relationship:

- Is supported by retrieved source text
- Does not hallucinate external information

Output: `True / False`


### 3. Retrieval Relevance

Checks whether the retrieved chunk:

- Shares keywords or semantic meaning with the query
- Is not completely unrelated

Output: `True / False`


## Output

For each chunk in Document B, the system produces:

- Query chunk
- Retrieved chunk
- Relationship answer
- Scores:
  - grounded
  - relevant
  - retrieval_relevant

Results are printed to the console for inspection.