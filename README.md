Rag agent witch takes one of the files as vector store and another one as query for similarity search. 
Documents are loaded and split into chunks.
Pairs of chunks with distance (cosine) <0.65 are trown away and the rest are classified as impacting, contradictign or with no relations by LLM.
Test results are provided in the repository.
