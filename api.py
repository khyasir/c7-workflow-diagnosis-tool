"""Backward-compatible alias.

The FastAPI app now lives in main.py so `uvicorn main:app` (Render's default
start command) works with no circular import. This module re-exports it so
`uvicorn api:app` keeps working too.
"""

from main import app  # noqa: F401

__all__ = ["app"]
