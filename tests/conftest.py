"""Skip model-dependent tests when the local Ollama server is not reachable.

The functional, security-guard and semantic-router tests invoke llama3 / phi3 through
Ollama. In environments without the models (CI, review sandboxes) they fail with a
connection error that says nothing about the code. They are skipped here with an explicit
reason instead; the evaluator and writer-fallback tests are pure Python and always run.
"""
import os
import socket
from urllib.parse import urlparse

import pytest

OLLAMA_DEPENDENT_FILES = {
    "test_functional_regression.py",
    "test_security_guard.py",
    "test_semantic_router.py",
}


def _ollama_reachable() -> bool:
    url = urlparse(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    host = url.hostname or "localhost"
    port = url.port or 11434
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if _ollama_reachable():
        return
    skip = pytest.mark.skip(reason="requires a running Ollama server with llama3/phi3; not reachable here")
    for item in items:
        if os.path.basename(str(item.fspath)) in OLLAMA_DEPENDENT_FILES:
            item.add_marker(skip)
