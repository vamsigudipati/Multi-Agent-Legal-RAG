from dotenv import load_dotenv
load_dotenv()  # Hooks up your .env file to the system for LangSmith tracking

from langgraph.graph import StateGraph, END
from state import LegalGraphState
from nodes import planner_node, executor_node, grader_node, replanner_node, writer_node

# ==========================================
# CONDITIONAL ROUTING LOGIC
# ==========================================

def check_checklist(state: LegalGraphState):
    """Router: Decides whether to start research or go straight to writing."""
    checklist = state.get("plan_checklist", [])
    if len(checklist) > 0:
        return "continue_research"
    return "finish_and_write"

def grader_router(state: LegalGraphState):
    """Router: Handles Grader pass/fail AND checks for remaining tasks."""
    # 1. First, check if the Grader flagged a failure
    if state.get("routing_critique"):
        return "replan"
    
    # 2. If it passed, check if we have more tasks to execute
    checklist = state.get("plan_checklist", [])
    if len(checklist) > 0:
        return "continue_research"
    
    # 3. If passed and checklist is empty, we are done!
    return "finish_and_write"

# ==========================================
# GRAPH CONSTRUCTION
# ==========================================

workflow = StateGraph(LegalGraphState)

# 1. Add our Agent Nodes
workflow.add_node("Planner", planner_node)
workflow.add_node("Executor", executor_node)
workflow.add_node("Grader", grader_node)
workflow.add_node("Replanner", replanner_node)
workflow.add_node("Writer", writer_node)

# 2. Define the Entry Point
workflow.set_entry_point("Planner")

# 3. Planner checks if there are tasks to execute
workflow.add_conditional_edges(
    "Planner",
    check_checklist,
    {
        "continue_research": "Executor",
        "finish_and_write": "Writer"
    }
)

# 4. Executor ALWAYS hands off to Grader for quality check
workflow.add_edge("Executor", "Grader")

# 5. Grader uses the Master Router to decide where to go next
workflow.add_conditional_edges(
    "Grader",
    grader_router,
    {
        "replan": "Replanner",
        "continue_research": "Executor",
        "finish_and_write": "Writer"
    }
)

# 6. Replanner ALWAYS loops back to Executor to run the new task
workflow.add_edge("Replanner", "Executor")

# 7. Writer either finishes or requests replanning based on evaluation
def writer_router(state: LegalGraphState):
    if state.get("routing_critique"):
        return "replan"
    return "finish"

workflow.add_conditional_edges(
    "Writer",
    writer_router,
    {
        "replan": "Replanner",
        "finish": END
    }
)

# Compile the Graph
app = workflow.compile()

# ==========================================
# EXECUTION (Local Testing)
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Local LangGraph Execution...\n")
    
    initial_state = {
        "user_query": "Assess whether 'garden leave' clauses combined with robust confidentiality and invention-assignment provisions are sufficient to protect a software company's intellectual property when hiring engineers who will work across California and New York. Provide model contract language (concise), state-specific compliance notes, and practical onboarding controls to minimize legal risk.",
        "last_query": "",
        "jurisdiction": "",
        "jurisdictions": [],
        "replan_count": 0,
        "plan_checklist": [],
        "completed_tasks": [],
        "failed_tasks": [],
        "retrieved_citations": [],
        "routing_critique": "",
        "final_briefing": ""
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n==========================================")
    print("🏆 FINAL LEGAL BRIEFING:")
    print("==========================================\n")
    print(final_state["final_briefing"])