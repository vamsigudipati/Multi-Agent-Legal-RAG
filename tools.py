from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import ValidationError
from state import LegalCitation
import os
import re
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


# ==========================================
# NEGATION-AWARE ENFORCEABILITY CHECK
# ==========================================
# Terms that mean the sentence is talking about a non-compete (directly or by reference).
_NONCOMPETE_TERMS = re.compile(
    r"non-?\s?compet(e|es|ition)|covenants?\s+not\s+to\s+compete|restrictive\s+covenants?",
    re.IGNORECASE,
)
# Anaphora that refer back to a non-compete already named in the briefing
# ("courts will not enforce them", "such clauses are enforceable").
_ANAPHORA = re.compile(
    r"\b(them|they|these|those|such|the\s+(clause|clauses|agreement|agreements|covenant|covenants|restriction|restrictions))\b",
    re.IGNORECASE,
)
# Affirmative enforceability language. "enforc" covers enforce/enforced/enforcing/enforceable
# but deliberately NOT unenforceable / non-enforceable (handled by the negators below).
_ENFORCE_CLAIM = re.compile(
    r"(?<![a-z-])(?<!un)enforc"
    r"|\buph(o|e)ld\b"
    r"|(?<!non-)(?<!not\s)\bbinding\b"
    r"|\bvalid\b"
    r"|increase\s+the\s+(likelihood|chance)"
    r"|sets?\s+a\s+precedent",
    re.IGNORECASE,
)
# Negation / invalidity markers. If any appears in the same clause as the claim,
# the clause is read as denying enforceability rather than asserting it.
_NEGATORS = re.compile(
    r"\b(not|no|never|cannot|can't|cant|won't|wont|don't|dont|doesn't|doesnt|isn't|isnt|aren't|arent"
    r"|neither|nor|unenforceable|non-enforceable|nonenforceable|void|voided|voids|invalid|invalidate[sd]?"
    r"|illegal|unlawful|prohibit(s|ed|ing)?|prohibition|ban|bans|banned|bar|bars|barred"
    r"|disfavor(s|ed|ing)?|outlaw(s|ed)?|struck\s+down|strikes?\s+down|refuse[sd]?\s+to|declin(e|es|ed)\s+to"
    r"|fail(s|ed)?\s+to|unlikely)\b"
    r"|n't\b",
    re.IGNORECASE,
)
# Other instruments a clause may be about instead of a non-compete. A clause that names one
# of these ("employers can enforce NDAs") is not charged against non-competes.
_OTHER_INSTRUMENTS = re.compile(
    r"\b(nda|ndas|non-?disclosure|confidentiality|non-?solicit\w*|invention[- ]assignment|trade[- ]secrets?"
    r"|garden[- ]leave|arbitration|severance|intellectual[- ]property|ip)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(
    r"[,;:()\u2014]|\s--\s|\b(but|however|although|though|whereas|while|unless|except|yet)\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _split_clauses(sentence: str) -> List[str]:
    parts = _CLAUSE_SPLIT.split(sentence)
    # re.split with a capturing group also returns the separators (or None); drop them.
    return [p.strip() for p in parts if p and p.strip() and not _CLAUSE_SPLIT.fullmatch(p.strip())]


def _clause_asserts_enforceability(clause: str) -> bool:
    """True when the clause claims enforceability and nothing in the clause negates it."""
    if not _ENFORCE_CLAIM.search(clause):
        return False
    return not _NEGATORS.search(clause)


def find_enforceability_contradictions(text: str) -> List[str]:
    """Return the sentences that affirmatively claim non-competes are enforceable.

    The text is split into sentences and each sentence into clauses. A clause is a
    contradiction when it contains enforceability language that is not negated within
    that same clause, and the clause is about non-competes: it names one, refers back to
    one ("courts will not enforce them"), or sits in a sentence about non-competes without
    naming some other instrument (an NDA, a non-solicit, an invention-assignment clause).

    Negated statements such as "non-competes are not enforceable in California",
    "courts will not enforce them", "void under §16600" and "California prohibits
    non-competes" are therefore left alone.
    """
    if not text or not _NONCOMPETE_TERMS.search(text):
        return []

    offending: List[str] = []
    for sentence in _split_sentences(text):
        sentence_is_about_noncompete = bool(_NONCOMPETE_TERMS.search(sentence))
        if not sentence_is_about_noncompete and not _ANAPHORA.search(sentence):
            continue
        for clause in _split_clauses(sentence):
            if not _clause_asserts_enforceability(clause):
                continue
            clause_names_noncompete = bool(_NONCOMPETE_TERMS.search(clause) or _ANAPHORA.search(clause))
            clause_names_other = bool(_OTHER_INSTRUMENTS.search(clause))
            if clause_names_noncompete or (sentence_is_about_noncompete and not clause_names_other):
                offending.append(sentence)
                break
    return offending


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
        # Negation-aware check: only sentences that affirmatively claim non-competes
        # are enforceable count as contradictions. "Non-competes are NOT enforceable
        # in California" is a correct statement and must not be flagged.
        offending = find_enforceability_contradictions(final_text)
        if offending:
            contradiction = True
            details.append("Contradiction: California Business & Professions Code §16600 disfavors employee non-competes.")
            for sentence in offending:
                details.append(f"Offending claim: \"{sentence[:160]}\"")
            score = min(score, 20)

    return {
        "score": score,
        "details": details,
        "contradiction": contradiction,
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }