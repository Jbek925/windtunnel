"""Guard against source files being silently excluded from git (this happened once)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout"
)
def test_no_source_file_is_gitignored() -> None:
    files = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("*.py")]
    out = subprocess.run(["git", "check-ignore", *files], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"these source files are gitignored:\n{out.stdout}"
