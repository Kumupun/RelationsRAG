from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def loader(file_path: str) :
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def chunking(doc_path: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    text = loader(doc_path)
    print(f"Document {doc_path} have been loaded successfully.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    return text_splitter.split_documents([Document(page_content=text, metadata={"source": doc_path})])