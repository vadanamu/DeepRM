import os
from pathlib import Path

def pytest_sessionstart(session):
    # Ensure the repository root is on PATH so the local `deeprm` script is found
    repo_root = Path(__file__).resolve().parent.parent
    os.environ["PATH"] = f"{repo_root}{os.pathsep}" + os.environ.get("PATH", "")
