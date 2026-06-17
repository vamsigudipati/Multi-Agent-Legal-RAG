import sys
import os
import pytest

# Ensure repo root is on sys.path so tests can import application modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import evaluate_briefing


def test_evaluator_flags_ca_noncompete_contradiction():
    final_text = (
        "Non-compete clauses for employees in California can be enforceable if the employer "
        "implements adequate protections and NDAs."
    )

    result = evaluate_briefing(final_text, jurisdiction="California")

    assert isinstance(result, dict)
    assert result.get("contradiction") is True, "Evaluator should flag a contradiction for CA non-compete claim"
    assert result.get("score", 100) < 50, "Score should be low when contradiction detected"


if __name__ == "__main__":
    pytest.main(["-q"]) 
