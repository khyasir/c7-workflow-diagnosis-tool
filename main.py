"""Workflow diagnosis tool — core logic + FastAPI app in one module.

diagnose() turns an analyst's workflow description into a buildable automation
plan via the Groq LLM. The FastAPI `app` is defined here too, so BOTH
`uvicorn main:app` (Render's default) and `uvicorn api:app` (thin alias) work
without a circular import.

Run the API:
    uvicorn main:app --reload
Run the CLI:
    python main.py "I export a CSV weekly, clean it in Excel, email a summary."
"""

import os
import sys
from typing import Any, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field, field_validator

# Pull GROQ_API_KEY out of the .env file and into the environment.
load_dotenv()

# The model we think with. 8b-instant is small + fast, plenty for a diagnosis.
MODEL = "llama-3.1-8b-instant"

# The whole "brain": tells the model its job is a build-able plan, not commentary.
SYSTEM_PROMPT = (
    "You are an automation diagnosis coach. The user is a data analyst who wants "
    "to start automating their work. Given their workflow description, return a "
    "short DIAGNOSIS PLAN with exactly these three parts:\n"
    "1. Repeatable steps: the manual, repeated steps you detect in their workflow.\n"
    "2. First automation: the single best thing to automate first "
    "(highest value, lowest effort) and why.\n"
    "3. Next steps: 3-5 concrete actions to build that first automation.\n"
    "Write plain text. Be specific to their workflow. No fluff."
)


def diagnose(workflow_description: str) -> str:
    """Turn a workflow description into a diagnosis plan using the Groq LLM.

    Args:
        workflow_description: free text describing what the analyst does.

    Returns:
        The diagnosis plan as plain text.
    """
    # Connect here (not at import time) so importing never fails on a missing key.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the .env file: GROQ_API_KEY=your_key"
        )

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": workflow_description},
            ],
        )
    except Exception as err:
        # Network down, bad key, rate limit, etc. — clean message, not a raw trace.
        raise RuntimeError(f"LLM call failed: {err}") from err

    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

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
            raise ValueError(
                "workflow_description must be text or a {'text': ...} object"
            )
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        description = input("Describe the workflow you want to automate:\n> ")

    if not description.strip():
        print("No workflow described. Nothing to diagnose.")
        sys.exit(1)

    print("\nDiagnosing...\n")
    try:
        print(diagnose(description))
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)
