import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
DATA_DIR = "./data"
DB_DIR = "./chroma_db"

def build_dynamic_database():
    print(f"📥 Loading documents from {DATA_DIR}...")
    
    # Loader mapping for different file types
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader
    }
    
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in loaders:
                print(f"  - Loading {file}...")
                loader = loaders[ext](os.path.join(root, file))
                documents.extend(loader.load())

    if not documents:
        print("⚠️ No documents found. Please add files to the ./data directory.")
        return

    print("✂️ Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    print("🧠 Embedding and saving to ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # We recreate the DB to ensure it matches the latest files
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"✅ Database built with {len(splits)} chunks.")

if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    build_dynamic_database()