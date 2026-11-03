"""The writer's replan-ceiling fallback must be honest: no canned answer, no perfect score.

These tests drive `writer_node` directly with a state at the ceiling. The reasoning LLM is
replaced with a stub that fails loudly if invoked, so no Ollama model is needed.
"""
import glob
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import nodes  # noqa: E402
from nodes import writer_node, MAX_REPLAN_ITERATIONS, FALLBACK_NO_PRECEDENTS_MSG  # noqa: E402

CANNED_MARKERS = [
    "compliant briefing requires careful multi-jurisdictional analysis",
    "PROHIBITIVE STATES (California, Oklahoma)",
    "REASONABLENESS STATES (New York, Texas)",
    "STATUTORY STATES (Washington, Illinois)",
    "Use a unified employment contract template",
]


class _ExplodingLLM:
    """Stand-in for the Ollama model: any call means the fallback path leaked into the LLM."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("LLM must not be invoked on the fallback path")

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


@pytest.fixture
def no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "reasoning_llm", _ExplodingLLM())
    monkeypatch.setattr(nodes, "admin_llm", _ExplodingLLM())
    monkeypatch.chdir(tmp_path)  # run JSON lands in tmp_path/runs
    return tmp_path


def _state(replan_count: int, citations=None) -> dict:
    return {
        "user_query": "Can a California employer enforce a non-compete against a software engineer?",
        "last_query": "",
        "jurisdiction": "California",
        "jurisdictions": ["California"],
        "replan_count": replan_count,
        "plan_checklist": [],
        "completed_tasks": [],
        "failed_tasks": [{"task": "Verify PROHIBITIVE enforcement status for California.", "error": "x"}],
        "retrieved_citations": citations or [],
        "routing_critique": "",
        "final_briefing": "",
        "security_rejection": "",
    }


def _latest_run(tmp_path) -> dict:
    files = sorted(glob.glob(os.path.join(tmp_path, "runs", "run_*.json")))
    assert files, "fallback must persist a run JSON"
    with open(files[-1]) as f:
        return json.load(f)


def test_ceiling_without_citations_returns_honest_no_precedents(no_model):
    out = writer_node(_state(replan_count=MAX_REPLAN_ITERATIONS))

    assert out["final_briefing"] == FALLBACK_NO_PRECEDENTS_MSG
    assert "unable to find specific legal precedents" in out["final_briefing"]
    for marker in CANNED_MARKERS:
        assert marker not in out["final_briefing"]

    ev = out["evaluation"]
    assert ev["score"] < 100
    assert ev["score"] == 0
    assert ev["fallback"] is True
    assert ev["fallback_reason"] == "max_replan_iterations_reached"
    assert not out.get("routing_critique"), "fallback must end the loop, not request another replan"

    run = _latest_run(no_model)
    assert run["fallback_reason"] == "max_replan_iterations_reached"
    assert run["evaluation"]["fallback"] is True


def test_ceiling_with_verified_citations_returns_labelled_partial(no_model):
    citations = [
        {"case_name": "Edwards v. Arthur Andersen LLP", "year": 2008, "jurisdiction": "California",
         "holding": "Employee non-competes are void under section 16600 and courts will not enforce them.",
         "is_statute": False},
        "Cal. Bus. & Prof. Code §16600: every contract restraining a lawful profession is to that extent void.",
    ]
    out = writer_node(_state(replan_count=MAX_REPLAN_ITERATIONS + 2, citations=citations))

    text = out["final_briefing"]
    assert text.startswith("PARTIAL BRIEFING (fallback)")
    assert "Edwards v. Arthur Andersen LLP" in text
    assert "§16600" in text
    for marker in CANNED_MARKERS:
        assert marker not in text

    ev = out["evaluation"]
    assert ev["fallback"] is True
    assert ev["fallback_reason"] == "max_replan_iterations_reached"
    assert ev["score"] < 100
    assert ev["score"] <= nodes.PARTIAL_BRIEFING_MAX_SCORE
    assert ev["contradiction"] is False, "verified negated citations must not be misread as a contradiction"
    assert not out.get("routing_critique")

    run = _latest_run(no_model)
    assert run["fallback_reason"] == "max_replan_iterations_reached"
    assert run["final_briefing"] == text


def test_below_ceiling_with_no_citations_uses_same_honest_message(no_model):
    out = writer_node(_state(replan_count=0))
    assert out["final_briefing"] == FALLBACK_NO_PRECEDENTS_MSG
    assert out["evaluation"]["fallback"] is True
    assert out["evaluation"]["fallback_reason"] == "empty_citations"
    assert out["evaluation"]["score"] == 0


def test_canned_summary_is_gone_from_source():
    with open(os.path.join(ROOT, "nodes.py")) as f:
        source = f.read()
    for marker in CANNED_MARKERS:
        assert marker not in source
