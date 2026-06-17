import sys
import os
import pytest

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import evaluate_briefing


def test_evaluator_flags_modal_ca_noncompete_contradiction():
    final_text = (
        "Based on the provided citations, non-compete clauses for software engineers in California may be enforceable if employers implement robust protections."
    )

    result = evaluate_briefing(final_text, jurisdiction="California")

    assert isinstance(result, dict)
    assert result.get("contradiction") is True
    assert result.get("score", 100) < 50


if __name__ == "__main__":
    pytest.main(["-q", os.path.basename(__file__)])
