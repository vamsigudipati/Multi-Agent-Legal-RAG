from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import ValidationError
from state import LegalCitation

# Initialize models
extractor_llm = ChatOllama(model="llama3", temperature=0.0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to the persistent local Chroma database
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Pull top 2 most relevant chunks

# Bind the Pydantic Guardrail
structured_llm = extractor_llm.with_structured_output(LegalCitation)

def query_chroma_db(query: str) -> str:
    """Production retrieval from the local vector database."""
    docs = retriever.invoke(query)
    # Combine the retrieved chunks into a single text block
    return "\n\n".join([doc.page_content for doc in docs])

def retrieve_and_validate(query: str) -> dict:
    """
    The Flow Engineering Loop: Retrieves real data and enforces the Pydantic schema.
    """
    # 1. Fetch REAL data from the vector database
    raw_text = query_chroma_db(query)
    
    # 2. Setup the strict extraction prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal data extractor. Extract the exact citation details from the provided source text to match the required schema. Do not invent information. If a detail is completely missing from the text, leave it blank."),
        ("user", "Source Text: {text}\n\nTask: Extract details related to: {query}")
    ])
    
    extraction_chain = prompt | structured_llm
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Attempt to extract data matching the Pydantic Guardrail
            result = extraction_chain.invoke({"text": raw_text, "query": query})
            
            return {
                "status": "success", 
                "data": result.model_dump()
            }
            
        except ValidationError as e:
            # Validation Failed - Triggering the Self-Correction Guardrail
            print(f"\n    🚨 PYDANTIC GUARDRAIL TRIGGERED on attempt {attempt + 1}")
            print(f"    Reason: {e.errors()[0]['msg']}")
            
            if attempt == max_retries - 1:
                return {
                    "status": "failed", 
                    "error": str(e),
                    "data": None
                }