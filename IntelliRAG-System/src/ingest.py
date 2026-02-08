import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_document(file_path):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Store with HuggingFace 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Ingested {len(texts)} chunks")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "_10-K-2025-As-Filed.pdf")
    
    if os.path.exists(file_path):
        ingest_document(file_path)
    else:
        print(f"Not found: {file_path}")