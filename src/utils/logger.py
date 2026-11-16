"""One logging setup for the whole project: console + logs/project.log."""
from __future__ import annotations

import logging
import sys

from src.constants import PROJECT_ROOT

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(console)
        try:
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "project.log", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(file_handler)
        except OSError:
            # A read-only folder must not stop training; console logging still works.
            root.warning("Could not create the log file; logging to console only.")
        _configured = True
    return logging.getLogger(name)
