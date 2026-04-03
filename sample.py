import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from config.settings import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

# -----------------------------
# 1. LOAD DOCUMENT
# -----------------------------


def load_documents(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# -----------------------------
# 2. SPLIT DOCUMENTS
# -----------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)


# -----------------------------
# 3. INIT PINECONE
# -----------------------------
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing_indexes = [index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    return pc


# -----------------------------
# 4. CREATE VECTOR STORE
# -----------------------------
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    return vector_store


# -----------------------------
# 5. INGEST PIPELINE
# -----------------------------
def run_ingestion(file_path: str):
    print("📄 Loading documents...")
    docs = load_documents(file_path)

    print(f"Loaded {len(docs)} documents")

    print("✂️ Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("🌲 Initializing Pinecone...")
    init_pinecone()

    print("🧠 Creating embeddings + storing in Pinecone...")
    create_vector_store(chunks)

    print("✅ Ingestion complete!")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    FILE_PATH = "src/data/sample.pdf"
    run_ingestion(FILE_PATH)
