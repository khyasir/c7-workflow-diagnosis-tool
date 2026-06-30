"""FastAPI wrapper around diagnose().

Exposes the workflow-diagnosis tool over HTTP so the student's automation can
call it from anywhere (web app, n8n, Zapier, curl).

Run it:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import diagnose

app = FastAPI(
    title="Workflow Diagnosis Tool",
    description="Turn an analyst's workflow description into a buildable automation plan.",
    version="1.0.0",
)


class DiagnoseRequest(BaseModel):
    workflow: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the analyst's workflow.",
        examples=[
            "Every Monday I export a CSV from Salesforce, clean it in Excel, "
            "and email a summary to my manager."
        ],
    )


class DiagnoseResponse(BaseModel):
    plan: str = Field(..., description="The diagnosis plan as plain text.")


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(req: DiagnoseRequest) -> DiagnoseResponse:
    """Generate a diagnosis plan from a workflow description."""
    if not req.workflow.strip():
        raise HTTPException(status_code=422, detail="workflow must not be empty")
    try:
        plan = diagnose(req.workflow)
    except RuntimeError as err:
        # Missing key / LLM failure -> 503, clean message instead of a 500 trace.
        raise HTTPException(status_code=503, detail=str(err)) from err
    return DiagnoseResponse(plan=plan)
