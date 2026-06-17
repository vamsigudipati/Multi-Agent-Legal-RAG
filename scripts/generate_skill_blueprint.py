#!/usr/bin/env python3
"""Simple CLI to create a skill blueprint from a short description.

Usage:
    python scripts/generate_skill_blueprint.py "Skill Name" "One-line description"

This is a lightweight helper that creates a markdown file under ./skills/.
"""
import sys
import os
from datetime import datetime, timezone

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: generate_skill_blueprint.py \"Skill Name\" \"Short description\"")
        sys.exit(1)

    name = sys.argv[1]
    desc = sys.argv[2]

    skills_dir = os.path.join(os.getcwd(), "skills")
    os.makedirs(skills_dir, exist_ok=True)

    safe_name = name.lower().replace(" ", "_")
    filename = f"{safe_name}.md"
    path = os.path.join(skills_dir, filename)

    now = datetime.now(timezone.utc).isoformat()

    content = f"""# {name}

    Description: {desc}

    Created: {now} UTC

    # Purpose

    - When to use: Short guidance.
    - Inputs: e.g., user query, jurisdiction hint
    - Outputs: e.g., citation markdown, structured JSON

    # Steps (example)

    1. Parse the query and extract jurisdictional constraints.
    2. Limit retrieval to jurisdiction-specific indexes.
    3. Extract holdings and cite with Bluebook-style citations.

    # Security Review

    - Reviewed: NO
    - Reviewer: TBD

    """

    with open(path, "w") as f:
        f.write(content)

    print(f"Created skill blueprint: {path}")
