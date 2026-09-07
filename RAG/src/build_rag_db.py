import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_databases():
    print("Loading lightweight embedding model...")
    # Uses HuggingFace embeddings which run locally without API costs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Reading RAGuard manuals (Technical and Safety)...")
    tech_docs = TextLoader("../data/tech_manuals.txt").load()
    safety_docs = TextLoader("../data/safety_manuals.txt").load()
    
    print("Building FAISS Vector Databases...")
    tech_db = FAISS.from_documents(tech_docs, embeddings)
    safety_db = FAISS.from_documents(safety_docs, embeddings)
    
    # Save the databases locally
    tech_db.save_local("../vector_store/tech_db")
    safety_db.save_local("../vector_store/safety_db")
    print("Success! RAG databases built and saved to /vector_store/")

if __name__ == "__main__":
    os.makedirs("../vector_store", exist_ok=True)
    build_vector_databases()