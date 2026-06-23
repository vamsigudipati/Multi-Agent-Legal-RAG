import sys
import os
# Add the root folder to Python's path so it can find nodes.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nodes import semantic_jurisdiction_router

def test_router_obfuscation_handling():
    """
    Tests if the few-shot prompt successfully translates slang/nicknames 
    into the official state name.
    """
    query = "Can my employer enforce a non-compete against me in the Golden State?"
    result = semantic_jurisdiction_router(query)
    
    assert "California" in result, f"Router failed to decode 'Golden State'. Got: {result}"

def test_router_unspecified_fallback():
    """
    Tests if the router strictly returns an empty list (Unspecified) 
    when no geographic location is mentioned, preventing hallucinated checklists.
    """
    query = "Based on the provided legal documents, what constitutes independent economic value?"
    result = semantic_jurisdiction_router(query)
    
    assert len(result) == 0, f"Router hallucinated a jurisdiction! Expected empty list, got: {result}"

def test_router_multiple_jurisdictions():
    """
    Tests if the router can accurately extract multiple explicitly named states.
    """
    query = "What are the core differences in employment law between NY and Texas?"
    result = semantic_jurisdiction_router(query)
    
    assert "New York" in result, f"Failed to extract New York. Got: {result}"
    assert "Texas" in result, f"Failed to extract Texas. Got: {result}"