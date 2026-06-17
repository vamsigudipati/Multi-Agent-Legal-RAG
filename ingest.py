import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Create a dummy legal document to act as our "Ground Truth" corpus
GROUND_TRUTH_FILE = "ground_truth_case_law.txt"
with open(GROUND_TRUTH_FILE, "w") as f:
    f.write("""
    IN THE SUPREME COURT OF NORTH CAROLINA
    Case Name: TechCorp v. Innovate Solutions
    Year: 2024
    Jurisdiction: North Carolina Supreme Court
    
    Background: TechCorp alleged that Innovate Solutions misappropriated proprietary 
    software architecture patterns related to agentic testing frameworks.
    
    Holding: The court ruled that software architecture patterns, when actively protected 
    by strict internal access controls and non-disclosure agreements, qualify as 
    trade secrets under the North Carolina Trade Secrets Protection Act. The court 
    emphasized that mere obscurity of the code is insufficient; affirmative protective 
    measures are required.
    """)

def build_vector_database():
    print("📥 Loading ground truth document...")
    loader = TextLoader(GROUND_TRUTH_FILE)
    docs = loader.load()

    print("✂️ Chunking text into processable segments...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("🧠 Generating embeddings and saving to ChromaDB...")
    # We use the local nomic-embed-text model pulled via Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Save the database locally to the ./chroma_db directory
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("✅ Production Database successfully built at ./chroma_db!")

if __name__ == "__main__":
    build_vector_database()