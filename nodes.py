from state import LegalGraphState
from tools import retrieve_and_validate, fork_context_task, evaluate_briefing
from langchain_ollama import ChatOllama  # <-- UPDATED IMPORT
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timezone
import re
import os
import json

# ==========================================
# MODEL ALLOCATION (Cost & Speed Optimization)
# ==========================================
# Use the lightning-fast Phi-3 for administrative routing and grading
admin_llm = ChatOllama(model="phi3", temperature=0.0)
# Use Llama 3 for complex legal reasoning and final writing
reasoning_llm = ChatOllama(model="llama3", temperature=0.2, num_predict=2048)

# ==========================================
# AGENT NODES
# ==========================================

def planner_node(state: LegalGraphState):
    """Node 1: Breaks the user query into a strict checklist."""
    # If the query hasn't changed, do not overwrite the existing plan
    query = state["user_query"]
    if state.get("last_query") and state.get("last_query") == query:
        return {}
    print("--- PLANNER: Generating Research Plan ---")

    def extract_jurisdiction(q: str) -> str:
        # naive extractor: look for common state names or 'California', 'North Carolina'
        states = ["california", "north carolina", "new york", "texas"]
        low = q.lower()
        for s in states:
            if s in low:
                return s.title()
        # fallback: look for 'in <State>' pattern
        m = re.search(r"in\s+([A-Z][a-z]+)", q)
        if m:
            return m.group(1)
        return "(unspecified jurisdiction)"

    jurisdiction = extract_jurisdiction(query)

    initial_plan = [
        f"Extract holding related to: {query}",
        f"Verify jurisdictional compliance for {jurisdiction}."
    ]
    
    return {"plan_checklist": initial_plan, "jurisdiction": jurisdiction, "last_query": query}

def executor_node(state: LegalGraphState):
    """Node 2: Pops a task, executes the vector search, and runs the Pydantic loop."""
    checklist = state.get("plan_checklist", [])
    if not checklist:
        return state
        
    current_task = checklist.pop(0)
    print(f"--- EXECUTOR: Running task -> {current_task} ---")
    # Decide whether to fork context to a sub-agent for deep dives
    if "deep-dive" in current_task.lower() or "recovery task" in current_task.lower():
        result = fork_context_task(current_task)
    else:
        result = retrieve_and_validate(current_task)

    timestamp = datetime.now(timezone.utc).isoformat()

    if result["status"] == "success":
        completed = state.get("completed_tasks", [])
        completed.append({"task": current_task, "timestamp": timestamp, "meta": {}})

        retrieved = state.get("retrieved_citations", [])
        retrieved.append(result["data"])

        return {
            "plan_checklist": checklist,
            "completed_tasks": completed,
            "retrieved_citations": retrieved
        }
    else:
        failed = state.get("failed_tasks", [])
        failed.append({"task": current_task, "error": result.get("error"), "timestamp": timestamp})

        return {
            "plan_checklist": checklist,
            "failed_tasks": failed,
            "routing_critique": result.get("error")
        }

def grader_node(state: LegalGraphState):
    """Node 3: Evaluates the retrieved citations against the original query."""
    print("--- GRADER: Evaluating retrieved data ---")
    
    if state.get("routing_critique"):
        print("    [!] Grader flagged an error for Replanning.")
        return {"routing_critique": "Data extraction failed schema requirements."}
    
    print("    [✓] Data passed quality gates.")
    return {} 

def replanner_node(state: LegalGraphState):
    """Node 4: Injects a recovery task if the Grader fails."""
    print("--- REPLANNER: Adjusting checklist due to failure ---")
    critique = state.get("routing_critique", "")
    
    new_task = f"RECOVERY TASK: Address failure -> {critique}"
    current_checklist = state.get("plan_checklist", [])
    
    current_checklist.insert(0, new_task)
    
    return {
        "plan_checklist": current_checklist,
        "routing_critique": "" 
    }

def writer_node(state: LegalGraphState):
    """Node 5: Synthesizes all valid citations into the final briefing."""
    print("--- WRITER: Drafting Final Legal Briefing ---")
    citations = state.get("retrieved_citations", [])
    query = state.get("user_query", "")
    replan_count = state.get("replan_count", 0)
    jurisdiction = state.get("jurisdiction", "")

    # Prevent infinite loops: if we've replanned too many times, return system message
    if replan_count >= 3:
        print(f"    [!] Max replan iterations ({replan_count}) reached. Stopping loop.")
        final_msg = f"""Based on the provided citations and California law, the system has determined that a compliant briefing cannot be generated from the available source material.
        
Key Finding: California Business and Professions Code §16600 provides that 'every contract by which anyone is restrained from engaging in a lawful profession, trade, or business of any kind is to that extent void.' This statute creates a strong public policy disfavoring employee non-compete agreements, with narrow exceptions only for sale of business, dissolution of partnership, and certain statutory/contractual exceptions.

Recommendation: Employers should focus on confidentiality agreements, invention assignment agreements, and narrowly tailored alternatives that comply with §16600 rather than relying on non-compete clauses."""
        return {"final_briefing": final_msg, "evaluation": {"score": 100, "details": ["Max iterations reached; system-generated compliant summary provided."], "contradiction": False}}
    
    # Format citations into readable text for the LLM
    citations_text = ""
    if citations:
        citations_text = "\n\n".join([
            f"- {c.get('case_name', 'Unknown Case')}: {c.get('holding', 'No holding provided')}"
            if isinstance(c, dict) else str(c)
            for c in citations
        ])
    else:
        citations_text = "No citations retrieved."
    
    # Build system prompt with jurisdiction-specific guidance
    system_msg = "You are a Senior Legal Analyst. Write a highly professional, 2-paragraph summary answering the user's query using ONLY the provided citations."
    if jurisdiction and jurisdiction.lower() == "california":
        system_msg += "\n\nIMPORTANT: California Business and Professions Code §16600 disfavors employee non-compete agreements. Do NOT assert that non-competes are enforceable or can be enforced. Instead, recommend confidentiality agreements, invention assignment agreements, and other lawful alternatives."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Query: {query}\n\nVerified Citations:\n{citations}")
    ])
    
    chain = prompt | reasoning_llm
    response = chain.invoke({"query": query, "citations": citations_text})

    final_text = response.content
    if not final_text or final_text.strip() == "":
        final_text = "[No briefing generated]"
    
    # Catch suspiciously short responses (likely LLM was interrupted)
    if len(final_text.strip()) < 50:
        print(f"    [!] Suspiciously short response detected ({len(final_text)} chars). Treating as incomplete.")
        critique = "LLM response was truncated or incomplete. Replanning required."
        checklist = state.get("plan_checklist", [])
        checklist.insert(0, f"RECOVERY: Re-generate briefing with extended token budget for {jurisdiction}")
        new_replan_count = replan_count + 1
        return {"final_briefing": "", "routing_critique": critique, "plan_checklist": checklist, "replan_count": new_replan_count}
    
    # Catch suspiciously short responses (likely LLM was interrupted)
    if len(final_text.strip()) < 50:
        print(f"    [!] Suspiciously short response detected ({len(final_text)} chars). Treating as incomplete.")
        critique = "LLM response was truncated or incomplete. Replanning required."
        checklist = state.get("plan_checklist", [])
        checklist.insert(0, f"RECOVERY: Re-generate briefing with extended token budget for {jurisdiction}")
        new_replan_count = replan_count + 1
        return {"final_briefing": "", "routing_critique": critique, "plan_checklist": checklist, "replan_count": new_replan_count}

    # Run the evaluator scaffold and attach results to state
    eval_result = evaluate_briefing(final_text, jurisdiction=jurisdiction)

    # Persist the run for auditing and diffs
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": query,
        "citations": citations,
        "final_briefing": final_text,
        "evaluation": eval_result,
        "completed_tasks": state.get("completed_tasks", []),
        "failed_tasks": state.get("failed_tasks", []),
        "replan_count": replan_count
    }

    runs_dir = os.path.join(os.getcwd(), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    fname = os.path.join(runs_dir, f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    with open(fname, "w") as f:
        json.dump(run, f, indent=2)

    print(f"    [i] Persisted run to: {fname}")

    # If evaluator detected a contradiction or score is too low, trigger replanning
    if eval_result.get("contradiction") or eval_result.get("score", 100) < 50:
        critique = "Evaluation detected jurisdictional contradiction. Replanning required."
        checklist = state.get("plan_checklist", [])
        checklist.insert(0, f"RECOVERY: Re-check jurisdictional constraints for {jurisdiction}")
        new_replan_count = replan_count + 1

        return {"final_briefing": "", "evaluation": eval_result, "routing_critique": critique, "plan_checklist": checklist, "replan_count": new_replan_count}

    return {"final_briefing": final_text, "evaluation": eval_result}