import sys
import os
import pytest

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import evaluate_briefing


def test_evaluator_flags_enforcing_phrase():
    final_text = (
        "Employers can minimize legal risk when enforcing non-compete clauses by implementing robust protections."
    )

    result = evaluate_briefing(final_text, jurisdiction="California")

    assert isinstance(result, dict)
    assert result.get("contradiction") is True
    assert result.get("score", 100) < 50


if __name__ == "__main__":
    pytest.main(["-q", os.path.basename(__file__)])
