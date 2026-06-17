# Contributing to Multi-Agent Legal RAG

Thanks for your interest in contributing. This document summarizes repository conventions, testing expectations, and security review requirements for `skills/` contributions.

## Getting started

1. Fork the repository and create a feature branch per change.
2. Keep changes small and focused; prefer multiple small PRs to a single large one.

## Development workflow

- Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Run tests:

```bash
pytest -q
```

- Run the pipeline locally for manual verification:

```bash
python ingest.py
python app.py
```

## Skills and Rule Files

- Add new skill blueprints under `skills/` using `scripts/generate_skill_blueprint.py`.
- Jurisdiction-specific guardrails belong in `rules/` with clear filenames like `rules/california_jurisdiction.md`.

## Security & Review Requirements

All contributed skills must include a `SECURITY_REVIEW.md` section in the same folder containing:

- Reviewer name and date
- Prompt injection risk assessment (yes/no) with brief notes
- External data dependency list
- Acceptance status (Reviewed / Needs changes)

Example header:

```
Reviewed: YES
Reviewer: Your Name
ReviewedAt: 2026-06-17
```

## Tests & Validation

- Unit tests should be added to `tests/` and must run locally with `pytest`.
- For extraction logic, maintain Pydantic schemas in `state.py` and ensure changes validate via the structured LLM output.

## Commit message guidelines

- Use conventional commits where practical (feat:, fix:, chore:, docs:, test:).
- Keep subject lines under 72 characters.

## Contact

If you need help, open an issue with the `help-wanted` tag or ping the repository owner.
