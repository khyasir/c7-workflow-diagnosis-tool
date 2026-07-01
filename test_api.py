"""Tests for the FastAPI app (defined in main.py).

Most tests stub diagnose() so they run without an API key and don't burn
tokens. One live test actually hits the LLM, gated on GROQ_API_KEY.

Run:  pytest test_api.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

import api
import main


def test_api_app_is_main_app():
    # api.py must re-export the same app object so `uvicorn api:app` still works.
    assert api.app is main.app


@pytest.fixture
def client(monkeypatch):
    """TestClient with diagnose() stubbed to a fixed plan."""
    monkeypatch.setattr(main, "diagnose", lambda wf: f"PLAN for: {wf}")
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_diagnose_success(client):
    r = client.post("/diagnose", json={"workflow_description": "I export CSVs weekly"})
    assert r.status_code == 200
    assert r.json() == {"plan": "PLAN for: I export CSVs weekly"}


def test_diagnose_multimodal_dict(client):
    # Gradio multimodal sends {"text":..., "files":[]} — backend must coerce it.
    r = client.post(
        "/diagnose",
        json={"workflow_description": {"text": "I export CSVs weekly", "files": []}},
    )
    assert r.status_code == 200
    assert r.json() == {"plan": "PLAN for: I export CSVs weekly"}


def test_diagnose_bad_type(client):
    # A list isn't text or a {text:...} dict -> 422.
    r = client.post("/diagnose", json={"workflow_description": [1, 2, 3]})
    assert r.status_code == 422


def test_diagnose_missing_field(client):
    # No 'workflow_description' key -> pydantic validation error.
    r = client.post("/diagnose", json={})
    assert r.status_code == 422


def test_diagnose_empty_string(client):
    # Empty string passes typing but the handler's .strip() guard returns 422.
    r = client.post("/diagnose", json={"workflow_description": ""})
    assert r.status_code == 422


def test_diagnose_whitespace_only(client):
    r = client.post("/diagnose", json={"workflow_description": "   "})
    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


def test_diagnose_llm_failure_returns_503(monkeypatch):
    # diagnose() raising RuntimeError (missing key / LLM down) -> 503, clean msg.
    def boom(_wf):
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(main, "diagnose", boom)
    client = TestClient(main.app)
    r = client.post("/diagnose", json={"workflow_description": "anything"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)
def test_diagnose_live():
    # Real end-to-end call. Skipped automatically when no key is present.
    client = TestClient(main.app)
    r = client.post(
        "/diagnose",
        json={
            "workflow_description": "Every Monday I export a CSV from Salesforce and email a summary."
        },
    )
    assert r.status_code == 200
    assert len(r.json()["plan"]) > 50
