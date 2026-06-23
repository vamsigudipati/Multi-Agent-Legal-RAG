import sys
import os
# Add the root folder to Python's path so it can find app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app import app

# Standard baseline state dictionary
def get_base_state(query: str) -> dict:
    return {
        "user_query": query,
        "last_query": "",
        "jurisdiction": "",
        "jurisdictions": [],
        "replan_count": 0,
        "plan_checklist": [],
        "completed_tasks": [],
        "failed_tasks": [],
        "retrieved_citations": [],
        "routing_critique": "",
        "final_briefing": "",
        "security_rejection": ""
    }

def test_circuit_breaker_and_anti_hallucination():
    """
    Tests the 'Eager Intern' fix: Ensures missing data trips the circuit breaker, 
    prevents infinite loops, and forces the safe fallback string without hallucinating.
    """
    # We use a query we know fails retrieval in the current database
    impossible_query = "Based on the provided legal documents, what constitutes 'independent economic value' when a court is determining whether a company's customer list qualifies as a protected trade secret?"
    state = get_base_state(impossible_query)
    
    # Run the graph
    final_state = app.invoke(state)
    
    # 1. CIRCUIT BREAKER ASSERTIONS
    assert len(final_state["failed_tasks"]) > 0, "Circuit breaker failed to log the dropped task."
    assert final_state["replan_count"] < 3, "System got stuck in a replan loop and didn't reset the counter."
    
    # 2. ANTI-HALLUCINATION ASSERTIONS
    expected_fallback = "I was unable to find specific legal precedents"
    assert expected_fallback in final_state["final_briefing"], "Writer failed to use the strict fallback string."
    
    # 3. NEGATIVE ASSERTIONS (Ensuring no hallucinations leaked)
    assert "Restatement of Torts" not in final_state["final_briefing"], "Writer hallucinated outside knowledge!"
    assert "Uniform Trade Secrets Act" not in final_state["final_briefing"], "Writer hallucinated outside knowledge!"

def test_successful_retrieval_and_routing():
    """
    Tests the 'Happy Path': Ensures a query that does exist in the database 
    successfully navigates the semantic grader and produces a briefing.
    """
    # Replace this string with a query you know successfully pulls data from your ChromaDB
    safe_retrieval_query = "What is the definition of a trade secret according to the provided documents?"
    state = get_base_state(safe_retrieval_query)
    
    final_state = app.invoke(state)
    
    # 1. ASSERT SUCCESSFUL EXECUTION
    assert final_state["security_rejection"] == "", "InputGuard falsely blocked a safe prompt."
    assert len(final_state["final_briefing"]) > 50, "Final briefing was empty or suspiciously short."
    
    # 2. ASSERT NO UNNECESSARY FAILURES
    assert "I was unable to find" not in final_state["final_briefing"], "System falsely claimed data was missing."