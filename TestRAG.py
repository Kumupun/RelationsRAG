from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature = 0)
embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/gemini-embedding-001",
    model = "text-embedding-004"
)

doc1_path = "Doc1.txt"
doc2_path = "Doc2.txt"

with open(doc1_path, "r", encoding="utf-8") as f:
    doc1_text = f.read()

with open(doc2_path, "r", encoding="utf-8") as f:
    doc2_text = f.read()

docs1 = [Document(page_content=doc1_text, metadata={"source": "doc1"})]
docs2 = [Document(page_content=doc2_text, metadata={"source": "doc2"})]

text_splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)
text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)
parag_split1 = text_splitter1.split_documents(docs1)
parag_split2 = text_splitter2.split_documents(docs2)

vectorstore = FAISS.from_documents(
    parag_split1,
    embeddings,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
)

def analyse_similarity():  
    chunks_similar = []
    for chunk in parag_split2:
        matches = vectorstore.similarity_search_with_score(
        chunk.page_content,
        k=2
    )
        for doc, score in matches:
                print(f"Score: {score} for chunk: {chunk.page_content[:50]}...")
                if score > 0.65:
                    chunks_similar.append((chunk, doc, score))
    return chunks_similar

prompt = ChatPromptTemplate.from_template("""
You are a logic analysis expert. Compare the following two document snippets.
Snippet 1 is from the Source Document (Context).
Snippet 2 is from the Query Document (New Info).

Determine their relationship using ONLY one of these labels:
- "Impacts": Snippet 1 causes, explains, or provides the direct reason for Snippet 2.
- "Contradicts": Snippet 1 and Snippet 2 contain mutually exclusive facts or conflicting data.
- "NO relationship": The snippets talk about different things or have no logical link.

Snippet 1: {doc1_chunk}
Snippet 2: {doc2_chunk}

Relationship Label:
Reasoning (in 1 sentence):""")
chain = prompt | llm 

def run_relationship_analysis():
    similar_pairs = analyse_similarity()
    
    print(f"Found {len(similar_pairs)} potential relationships. Analyzing...\n")
    
    for query_chunk, source_doc, score in similar_pairs:
        relation = chain.invoke({
            "doc1_chunk": source_doc.page_content,
            "doc2_chunk": query_chunk.page_content
        })
        
        print(f"MATCH (Score: {score:.2f}):")
        print(f"Source Snippet: \n{source_doc.page_content}")
        print(f"Query Snippet: \n{query_chunk.page_content}")
        print(f"VERDICT:\n{relation.content}\n")


run_relationship_analysis()