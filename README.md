# RAG Pipeline with Threshold\Prompt Optimization

A modular Retrieval-Augmented Generation system with threshold tuning and LLM-as-Judge evaluation tracked via Weights & Biases.

## Overview
This project implements a synchronous RAG pipeline with:

- FAISS-based dense retrieval
- Structured LLM generation
- LLM-as-Judge evaluation
- Prompt optimization
- Threshold tuning
- W&B experiment tracking

## Architecture

### Data Processing
- Document loading
- Recursive Character Text Splitter
- Embeddings (nomic-embed-text)

### Retrieval
- FAISS vector store
- Scoring similarity search

### Generation
- Structured prompt format
- Deterministic output constraints

### Evaluation
- LLM-as-Judge
- Strict boolean metrics grading


### Threshold tuning
- Precision
- Recall
- F1-score
- Weighted precision-recall score
- W&B logging
  
## Full Pipeline Execution
This script runs the complete RAG experiment:
1. Document chunking
2. FAISS vector indexing
3. Similarity-based retrieval
4. LLM semantic relationship judgment
5. Structured evaluation
6. Threshold optimization with W&B logging

/assets/W&B Chart Best threshold.png
/assets/W&B Chart F1 score.png
/assets/W&B Chart Precision.png
/assets/W&B Chart Recall.png
/assets/W&B Chart Weighted score.png
