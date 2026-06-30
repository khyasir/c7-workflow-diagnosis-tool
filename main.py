"""Workflow diagnosis tool.

Input:  a free-text description of an analyst's workflow.
Output: a diagnosis PLAN they can follow to build their first automation.

Run it:
    python main.py "I download a CSV from Salesforce every Monday, clean it in Excel, email a summary."
or just `python main.py` and type the description when asked.
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

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
