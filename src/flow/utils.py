"""
utils.py — utilidades.
"""
from pathlib import Path

def ensure_dirs(root: Path, subfolders: list[str]) -> None:
    for sub in subfolders:
        (root / sub).mkdir(parents=True, exist_ok=True)
