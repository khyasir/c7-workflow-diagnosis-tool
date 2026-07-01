"""FastAPI wrapper around diagnose().

Exposes the workflow-diagnosis tool over HTTP so the student's automation can
call it from anywhere (web app, n8n, Zapier, curl).

Run it:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from typing import Any, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from main import diagnose

app = FastAPI(
    title="Workflow Diagnosis Tool",
    description="Turn an analyst's workflow description into a buildable automation plan.",
    version="1.0.0",
)


class DiagnoseRequest(BaseModel):
    # Accept either a plain string or Gradio's multimodal dict ({"text": ...,
    # "files": [...]}) — clients vary, so normalize to a string before validating.
    workflow_description: Union[str, dict] = Field(
        ...,
        description="Workflow text, or a {'text': ...} object from a chat client.",
        examples=[
            "Every Monday I export a CSV from Salesforce, clean it in Excel, "
            "and email a summary to my manager."
        ],
    )

    @field_validator("workflow_description", mode="before")
    @classmethod
    def coerce_to_text(cls, v: Any) -> str:
        if isinstance(v, dict):
            v = v.get("text", "")  # Gradio multimodal shape
        if not isinstance(v, str):
            raise ValueError("workflow_description must be text or a {'text': ...} object")
        return v


class DiagnoseResponse(BaseModel):
    plan: str = Field(..., description="The diagnosis plan as plain text.")


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(req: DiagnoseRequest) -> DiagnoseResponse:
    """Generate a diagnosis plan from a workflow description."""
    if not req.workflow_description.strip():
        raise HTTPException(
            status_code=422, detail="workflow_description must not be empty"
        )
    try:
        plan = diagnose(req.workflow_description)
    except RuntimeError as err:
        # Missing key / LLM failure -> 503, clean message instead of a 500 trace.
        raise HTTPException(status_code=503, detail=str(err)) from err
    return DiagnoseResponse(plan=plan)
