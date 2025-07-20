from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import os

DATA_DIR = "../knowledge_base_docs"
CHROMA_DB_DIR = "../backend/vector_db"

def load_documents():
    loaders = [
        DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyMuPDFLoader),
        DirectoryLoader(DATA_DIR, glob="*.md", loader_cls=TextLoader)
    ]
    all_docs = []
    for loader in loaders:
        all_docs.extend(loader.load())
    return all_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def split_docs(docs):
    return text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    print("Loading and splitting documents...")
    docs = load_documents()
    chunks = split_docs(docs)
    print(f"Loaded {len(chunks)} chunks")

    print("Building Chroma vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectordb.persist()
    print("Vector store saved.")

if __name__ == "__main__":
    build_vector_store()
