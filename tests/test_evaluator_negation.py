"""Negation-aware contradiction check for California non-compete claims.

These tests exercise `evaluate_briefing` / `find_enforceability_contradictions` directly.
They need no Ollama model: the evaluator is pure Python.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import evaluate_briefing, find_enforceability_contradictions  # noqa: E402


def _is_flagged(text: str) -> bool:
    result = evaluate_briefing(text, jurisdiction="California")
    assert set(result) >= {"score", "details", "contradiction", "evaluated_at"}, "return shape changed"
    if result["contradiction"]:
        assert result["score"] < 50
    return result["contradiction"]


# ---------------------------------------------------------------------------
# True contradictions: affirmative enforceability claims must still be caught
# ---------------------------------------------------------------------------
TRUE_CONTRADICTIONS = [
    "Non-competes are enforceable in California if reasonable.",
    "Non-compete clauses for employees in California can be enforceable if the employer implements adequate protections and NDAs.",
    "Based on the provided citations, non-compete clauses for software engineers in California may be enforceable if employers implement robust protections.",
    "Employers can minimize legal risk when enforcing non-compete clauses by implementing robust protections.",
    "California courts routinely enforce non-competes that are reasonable in scope and duration.",
    "A well-drafted non-compete is valid and binding on California employees.",
    "Non-competes are enforceable in California, not void.",
]


@pytest.mark.parametrize("text", TRUE_CONTRADICTIONS)
def test_true_contradictions_are_caught(text):
    assert _is_flagged(text), f"Should be flagged: {text}"


# ---------------------------------------------------------------------------
# Correct, negated statements must NOT be flagged
# ---------------------------------------------------------------------------
CORRECT_STATEMENTS = [
    "Non-competes are not enforceable in California.",
    "Employee non-competes are void in California, and courts will not enforce them.",
    "Courts may not enforce non-competes against California employees.",
    "Non-competes cannot be enforced in California.",
    "Non-compete agreements are unenforceable in California under Business and Professions Code section 16600.",
    "Employee non-compete clauses are void under §16600.",
    "California prohibits non-competes.",
    "California courts never uphold non-competes signed by rank-and-file employees.",
    "Non-competes are not enforceable in California, even if they are narrowly tailored and reasonable.",
    "Non-competes are prohibited; employers should rely on confidentiality agreements instead.",
]


@pytest.mark.parametrize("text", CORRECT_STATEMENTS)
def test_negated_statements_are_not_flagged(text):
    assert not _is_flagged(text), f"False positive: {text}"


def test_no_noncompete_mention_is_not_flagged():
    text = "Trade secret protections and confidentiality agreements are enforceable in California."
    assert not _is_flagged(text)


# ---------------------------------------------------------------------------
# Mixed paragraphs
# ---------------------------------------------------------------------------
def test_mixed_paragraph_correct_is_not_flagged():
    text = (
        "Under Business and Professions Code §16600, employee non-competes are void in California. "
        "Courts will not enforce them, and the Legislature prohibits employers from requiring them. "
        "Although non-competes are generally not enforceable in California, employers can enforce "
        "NDAs and invention assignment agreements to protect trade secrets. "
        "Recommendation: use confidentiality and non-solicitation provisions instead."
    )
    assert find_enforceability_contradictions(text) == []
    assert not _is_flagged(text)


def test_mixed_paragraph_with_buried_contradiction_is_flagged():
    text = (
        "California prohibits most employee non-competes, and §16600 renders them void. "
        "However, a narrowly drafted non-compete is enforceable if the employee is a senior executive. "
        "Employers should still prefer confidentiality agreements."
    )
    offending = find_enforceability_contradictions(text)
    assert len(offending) == 1
    assert "narrowly drafted non-compete is enforceable" in offending[0]
    assert _is_flagged(text)


def test_details_name_the_offending_sentence():
    text = "Non-competes are void under §16600. Courts nonetheless enforce non-competes against executives."
    result = evaluate_briefing(text, jurisdiction="California")
    assert result["contradiction"] is True
    assert any("Courts nonetheless enforce non-competes" in d for d in result["details"])


def test_non_california_jurisdiction_skips_the_check():
    result = evaluate_briefing("Non-competes are enforceable in Texas if reasonable.", jurisdiction="Texas")
    assert result["contradiction"] is False
