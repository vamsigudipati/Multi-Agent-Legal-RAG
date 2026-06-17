from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import ValidationError
from state import LegalCitation
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Initialize models
extractor_llm = ChatOllama(model="llama3", temperature=0.0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to the persistent local Chroma database
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Pull top 2 most relevant chunks

# Bind the Pydantic Guardrail
structured_llm = extractor_llm.with_structured_output(LegalCitation)

def query_chroma_db(query: str, path_scope: Optional[str] = None, k: int = 2) -> str:
    """Production retrieval from the local vector database with optional path scoping."""
    # Allow overriding the number of docs for deep dives
    docs = retriever.invoke(query)

    # If a path_scope is provided try to filter by the source metadata
    if path_scope:
        filtered = [d for d in docs if getattr(d, "metadata", {}).get("source", "").find(path_scope) != -1]
        if filtered:
            docs = filtered

    # Combine the retrieved chunks into a single text block
    return "\n\n".join([doc.page_content for doc in docs[:k]])

def retrieve_and_validate(query: str, path_scope: Optional[str] = None, k: int = 2) -> dict:
    """
    The Flow Engineering Loop: Retrieves real data and enforces the Pydantic schema.
    """
    # 1. Fetch REAL data from the vector database
    raw_text = query_chroma_db(query, path_scope=path_scope, k=k)
    
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


def load_rules_for_task(task: str) -> Optional[str]:
    """Load a path-specific rule filename for a given task.

    Simple heuristic: if the word 'jurisdiction' or a state name appears,
    return an appropriate rule filename under ./rules/ if present.
    """
    rules_dir = os.path.join(os.getcwd(), "rules")
    if not os.path.isdir(rules_dir):
        return None

    # naive match
    lowered = task.lower()
    for fname in os.listdir(rules_dir):
        if fname.lower().replace(".md", "") in lowered:
            return os.path.join(rules_dir, fname)

    # fallback: look for jurisdiction guardrail files
    for fname in os.listdir(rules_dir):
        if "jurisdiction" in fname.lower():
            return os.path.join(rules_dir, fname)

    return None


def fork_context_task(task: str) -> Dict[str, Any]:
    """Run a scoped retrieval as a 'sub-agent' using a narrower context.

    Returns same dict shape as `retrieve_and_validate`.
    """
    # Try to detect a path or jurisdiction to scope to
    rule_file = load_rules_for_task(task)
    path_scope = None
    if rule_file:
        # if rule exists, use its filename (without extension) as a hint
        path_scope = os.path.splitext(os.path.basename(rule_file))[0]

    # For deep dives, increase k
    return retrieve_and_validate(task, path_scope=path_scope, k=5)


def evaluate_briefing(final_text: str, jurisdiction: Optional[str] = None, gold: Optional[str] = None) -> Dict[str, Any]:
    """Evaluator scaffold: compares final briefing to optional gold and jurisdictional constraints.

    Returns rubric with numeric score and a `contradiction` flag when the briefing conflicts
    with known jurisdictional rules (e.g., California's §16600 prohibition on employee non-competes).
    """
    score = 90
    details: List[str] = []
    contradiction = False

    # Load canonical gold for jurisdiction when available
    if not gold and jurisdiction and jurisdiction.lower() == "california":
        # Load gold text relative to this module so it doesn't depend on cwd
        module_dir = os.path.dirname(os.path.abspath(__file__))
        gold_path = os.path.join(module_dir, "rules", "gold_ca_16600.txt")
        try:
            with open(gold_path, "r") as gf:
                gold = gf.read()
                details.append("Loaded canonical gold text for California (§16600).")
        except FileNotFoundError as e:
            details.append(f"No canonical gold found at {gold_path}; proceeding without it. Error: {e}")
            print(f"[DEBUG] Gold file not found. Checked: {gold_path}", file=__import__('sys').stderr)

    # Gold comparison (exact match heuristic)
    if gold:
        if gold.strip() == final_text.strip():
            score = 100
            details.append("Exact match to gold standard.")
        else:
            details.append("Non-identical to gold; manual review recommended.")
            score = min(score, 70)
    else:
        details.append("No gold provided — used heuristic checks.")

    # Jurisdiction-specific heuristics
    if jurisdiction and jurisdiction.lower() == "california":
        # If the briefing asserts that non-competes are enforceable in CA, that's a contradiction.
        # Detect direct claims and modal language that implies enforceability.
        lower = final_text.lower()
        has_noncomp = "non-compete" in lower or "noncompete" in lower


        # Use a stem to catch enforce/enforcing/enforced/enforceable and modal phrasing
        enforce_patterns = [
            r"enforc",  # stem match for enforce/enforcing/enforced/enforceable
            r"can be enforc",
            r"may be enforc",
            r"could be enforc",
            r"likely.*enforc",
            r"increase the likelihood",
            r"increase the chance",
            r"minimiz.*legal risk.*enforc",
            r"sets a precedent for enforc",  # catch "sets a precedent for enforcing"
            r"can effectively enforc"
        ]

        import re
        enforces = any(re.search(pat, lower) for pat in enforce_patterns)

        if has_noncomp and enforces:
            contradiction = True
            details.append("Contradiction: California Business & Professions Code §16600 disfavors employee non-competes.")
            score = min(score, 20)

    return {
        "score": score,
        "details": details,
        "contradiction": contradiction,
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }