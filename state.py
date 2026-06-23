from typing import TypedDict, List, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field
import operator

# ==========================================
# 1. PYDANTIC v2 VALIDATION CONTRACTS
# ==========================================
# This schema is the "Hard Gate" for our Flow Engineering loop. 
# If the LLM misses any of these fields during retrieval, Pydantic throws 
# a ValidationError, triggering the Retriever to auto-correct.

class LegalCitation(BaseModel):
    """Schema representing a verified legal case extraction."""
    case_name: str = Field(description="The formal name of the case (e.g., Smith v. Jones)")
    year: int = Field(description="The year the opinion was published")
    jurisdiction: str = Field(description="The court or state jurisdiction (e.g., North Carolina)")
    holding: str = Field(description="The core legal ruling or extraction answering the query")
    is_statute: bool = Field(description="True if this is a codified statute, False if case law")

# ==========================================
# 2. LANGGRAPH SHARED STATE (TypedDict)
# ==========================================
# This is the central dashboard passed between all nodes.

class LegalGraphState(TypedDict):
    # The original request from the user
    user_query: str
    
    # In state.py, add this key to your TypedDict:
    security_rejection: str
    
    # Track which query the plan was generated for (avoid re-planning on same query)
    last_query: str
    
    # Extracted primary jurisdiction for rule scoping
    jurisdiction: str
    
    # All extracted jurisdictions (for multi-state analysis)
    jurisdictions: List[str]
    
    # Track replanning iterations to prevent infinite loops
    replan_count: int
    
    # The dynamic checklist managed by the Planner and Replanner
    plan_checklist: List[str]
    current_task: str
    current_context: str
    
    # Audit trail: Keep track of what we've done
    # Each entry is a structured record to enable analytics
    completed_tasks: Annotated[List[Dict[str, Any]], operator.add]
    failed_tasks: Annotated[List[Dict[str, Any]], operator.add]
    
    # The verified data pulled by the Executor/Retriever
    retrieved_citations: Annotated[List[Dict[str, Any]], operator.add]
    
    # The communication channel from the Grader to the Replanner
    routing_critique: str
    
    # The final output compiled by the Writer
    final_briefing: str