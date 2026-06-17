from state import LegalGraphState
from tools import retrieve_and_validate
from langchain_ollama import ChatOllama  # <-- UPDATED IMPORT
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# MODEL ALLOCATION (Cost & Speed Optimization)
# ==========================================
# Use the lightning-fast Phi-3 for administrative routing and grading
admin_llm = ChatOllama(model="phi3", temperature=0.0)
# Use Llama 3 for complex legal reasoning and final writing
reasoning_llm = ChatOllama(model="llama3", temperature=0.2)

# ==========================================
# AGENT NODES
# ==========================================

def planner_node(state: LegalGraphState):
    """Node 1: Breaks the user query into a strict checklist."""
    # Check if a plan already exists! If so, DO NOT overwrite it.
    if state.get("plan_checklist"):
        return {} # Return empty dict to leave state unchanged
        
    query = state["user_query"]
    
    print("--- PLANNER: Generating Research Plan ---")
    initial_plan = [
        f"Extract holding related to: {query}",
        "Verify jurisdictional compliance for North Carolina."
    ]
    
    return {"plan_checklist": initial_plan}

def executor_node(state: LegalGraphState):
    """Node 2: Pops a task, executes the vector search, and runs the Pydantic loop."""
    checklist = state.get("plan_checklist", [])
    if not checklist:
        return state
        
    current_task = checklist.pop(0)
    print(f"--- EXECUTOR: Running task -> {current_task} ---")
    
    result = retrieve_and_validate(current_task)
    
    if result["status"] == "success":
        return {
            "plan_checklist": checklist,
            "retrieved_citations": [result["data"]]
        }
    else:
        return {
            "plan_checklist": checklist,
            "failed_tasks": [current_task],
            "routing_critique": result["error"]
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Legal Analyst. Write a highly professional, 2-paragraph summary answering the user's query using ONLY the provided citations."),
        ("user", "Query: {query}\n\nVerified Citations: {citations}")
    ])
    
    chain = prompt | reasoning_llm
    response = chain.invoke({"query": query, "citations": citations})
    
    return {"final_briefing": response.content}