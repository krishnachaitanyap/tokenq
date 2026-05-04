"""No-install launcher.

Adjusts sys.path so the source tree is importable without `pip install`,
then forwards every argument to the Typer CLI. Use exactly like the
`tokenq` console script:

    python run.py start              # proxy + dashboard + MCP (multi-process)
    python run.py serve              # all three in one process
    python run.py status
    python run.py stop

Useful on locked-down environments where pip install is restricted but
running Python from an approved location is fine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tokenq.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
