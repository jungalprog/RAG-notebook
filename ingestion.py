import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader


from logger import (Colors, log_error, log_header, log_info)


load_dotenv()

# Embed docs
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              show_progress_bar=False,
                              chunk_size=50,
                              retry_min_seconds=10)

# Pinecone
vectorstore = PineconeVectorStore(
    index_name="rag-notebook", embedding=embeddings)


# load docs
def load_documents(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# Split texts into chunks
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_vectorstore(chunks: str):
    try:
        vectorstore.add_documents(chunks)
        log_info(
            f"VectorStore Indexing successful! Added {len(chunks)} document")
    except Exception as e:
        log_error(f"VectorStore indexing failed: {e}")
        return False
    return True


def run_ingestion(file_path: str):
    log_header("🚀 Initiating Document Ingestion")
    docs = load_documents(file_path)

    log_info(f"Loaded {len(docs)} documents", Colors.CYAN,)

    log_header("Splitting documents...")
    chunks = split_docs(docs)
    log_info(f"Created {len(chunks)} documents", Colors.RED,)

    log_header("Indexing to VectorStore...")
    create_vectorstore(chunks)
    print("Ingestion Complete!")
    log_header("Ingestion Complete!")


# splitted_docs = text_splitter.split_documents(all_docs)
if __name__ == '__main__':
    FILE_PATH = "src/Meal Plan.pdf"
    run_ingestion(FILE_PATH)
