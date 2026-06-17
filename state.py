from typing import TypedDict, List, Dict, Any, Annotated
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
    
    # The dynamic checklist managed by the Planner and Replanner
    plan_checklist: List[str]
    
    # Audit trail: Keep track of what we've done
    completed_tasks: Annotated[List[str], operator.add]
    failed_tasks: Annotated[List[str], operator.add]
    
    # The verified data pulled by the Executor/Retriever
    retrieved_citations: Annotated[List[Dict[str, Any]], operator.add]
    
    # The communication channel from the Grader to the Replanner
    routing_critique: str
    
    # The final output compiled by the Writer
    final_briefing: str