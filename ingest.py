import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

def load_and_split_pdf(pdf_path: str):
    """Load PDF and split into chunks"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Loaded {len(pages)} pages → {len(chunks)} chunks")
    return chunks

def store_in_chromadb(chunks, collection_name="pdf_docs"):
    """Store chunks as embeddings in ChromaDB"""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def load_vectorstore(collection_name="pdf_docs"):
    """Load existing ChromaDB vectorstore"""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore

def ingest_pdf(pdf_path: str):
    """Full pipeline: PDF → chunks → ChromaDB"""
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = store_in_chromadb(chunks)
    return vectorstore