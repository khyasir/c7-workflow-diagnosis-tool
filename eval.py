"""Eval for the diagnose() function.

Inspired by the Ogilvy copy-scorer: instead of grading marketing copy, this
grades a DIAGNOSIS PLAN produced by diagnose(). An LLM judge scores the plan
out of 100 across 15 automation-diagnosis principles (~6.7 pts each), explains
each score, names the top 3 fixes, and rewrites the plan to score 100/100.

Run it:
    python eval.py "I download a CSV from Salesforce every Monday, clean it in Excel, email a summary."
or pipe a plan you already have:
    python eval.py --plan-file plan.txt --workflow "the original workflow text"
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

from main import diagnose

load_dotenv()

# Judge can use a bigger model than the generator — better at scoring/critique.
JUDGE_MODEL = "llama-3.3-70b-versatile"

# The 15 principles. Each scored 0–6.7 → 100.5 max, clamped to 100.
# Adapted from Ogilvy's copy principles to what makes a diagnosis plan buildable.
PRINCIPLES = [
    ("Step Detection", "Correctly identifies the manual, repeated steps in the workflow."),
    ("Specificity", "Specific to THIS workflow, not generic automation advice."),
    ("First-Automation Choice", "Picks the highest-value, lowest-effort thing to automate first."),
    ("Justification", "Explains WHY that first automation wins (value vs effort)."),
    ("Actionability", "Next steps are concrete and the analyst can act on them today."),
    ("Sequencing", "Steps are in a sensible build order, no skipped prerequisites."),
    ("Tooling", "Names real, appropriate tools/tech for the job."),
    ("Clarity", "Plain language, no unexplained jargon."),
    ("Completeness", "Covers all 3 required parts: steps, first automation, next steps."),
    ("Feasibility", "Realistic for a data analyst's skill level and tools."),
    ("Reader-Fit", "Centered on the analyst's needs, not abstract best practice."),
    ("Measurability", "Quantifies payoff (time saved, errors cut) where possible."),
    ("Structure", "Skimmable and well-formatted."),
    ("Risk Awareness", "Flags pitfalls, edge cases, or what could break."),
    ("Motivation", "Hooks the analyst — makes starting feel worth it."),
]

POINTS_EACH = 100 / len(PRINCIPLES)  # ~6.67

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of automation DIAGNOSIS PLANS, trained on "
    "David Ogilvy's principles of clarity, specificity, and proof. A diagnosis "
    "plan is given to a data analyst so they can build their first automation.\n\n"
    "You will receive the analyst's WORKFLOW and the PLAN produced for them. "
    f"Score the PLAN against these {len(PRINCIPLES)} principles, each out of "
    f"{POINTS_EACH:.2f} points:\n"
    + "\n".join(f"{i+1}. {name}: {desc}" for i, (name, desc) in enumerate(PRINCIPLES))
    + "\n\nBe harsh. Reserve high scores for plans that are specific, buildable, "
    "and proof-driven. Return ONLY valid JSON, no prose, with this shape:\n"
    "{\n"
    '  "scores": [{"principle": str, "score": number, "comment": str}, ...],\n'
    '  "top_3_improvements": [str, str, str],\n'
    '  "suggested_edits": [str, ...],\n'
    '  "rewrite": str   // the plan rewritten to score 100/100\n'
    "}\n"
    f"Provide exactly {len(PRINCIPLES)} score objects in principle order. "
    f"Each score is 0 to {POINTS_EACH:.2f}."
)


def _judge(workflow_description: str, plan: str) -> dict:
    """Ask the judge LLM to score a plan. Returns parsed JSON dict."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the .env file: GROQ_API_KEY=your_key"
        )

    client = Groq(api_key=api_key)
    user_msg = f"WORKFLOW:\n{workflow_description}\n\nPLAN:\n{plan}"

    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=2048,
            temperature=0,  # deterministic scoring
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as err:
        raise RuntimeError(f"Judge LLM call failed: {err}") from err

    raw = completion.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Judge returned non-JSON output: {err}\n{raw}") from err


def evaluate(workflow_description: str, plan: str | None = None) -> dict:
    """Generate a plan (if not supplied), judge it, and return the full result.

    Args:
        workflow_description: the analyst's workflow.
        plan: an existing plan to grade; if None, diagnose() produces one.

    Returns:
        dict with keys: workflow, plan, overall, scores, top_3_improvements,
        suggested_edits, rewrite.
    """
    if plan is None:
        plan = diagnose(workflow_description)

    judged = _judge(workflow_description, plan)

    # Clamp each score to its band, then sum and clamp the total to 100.
    for s in judged.get("scores", []):
        s["score"] = max(0.0, min(POINTS_EACH, float(s.get("score", 0))))
    overall = min(100.0, sum(s["score"] for s in judged.get("scores", [])))

    return {
        "workflow": workflow_description,
        "plan": plan,
        "overall": round(overall, 1),
        "scores": judged.get("scores", []),
        "top_3_improvements": judged.get("top_3_improvements", []),
        "suggested_edits": judged.get("suggested_edits", []),
        "rewrite": judged.get("rewrite", ""),
    }


def format_report(result: dict) -> str:
    """Render the evaluation as an Ogilvy-style Markdown report."""
    lines = [
        f"**Workflow Analyzed:** {result['workflow']}",
        "",
        f"**Overall Score:** {result['overall']}/100",
        "",
        "**Score Breakdown:**",
        "",
        f"| Principle | Score (0–{POINTS_EACH:.1f}) | Comments |",
        "|-----------|------------|----------|",
    ]
    for i, s in enumerate(result["scores"], 1):
        name = s.get("principle", f"#{i}")
        score = s.get("score", 0)
        comment = str(s.get("comment", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i}. {name} | {score:.1f} | {comment} |")

    lines += ["", "**Top 3 Areas to Improve:**"]
    for i, item in enumerate(result["top_3_improvements"][:3], 1):
        lines.append(f"{i}. {item}")

    if result["suggested_edits"]:
        lines += ["", "**Suggested Edits:**"]
        lines += [f"- {e}" for e in result["suggested_edits"]]

    lines += ["", "---", "", "### Rewrite (to score 100/100):", "", result["rewrite"]]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval the diagnose() output.")
    parser.add_argument("workflow", nargs="*", help="workflow description text")
    parser.add_argument("--workflow", dest="workflow_opt", help="workflow text (alt)")
    parser.add_argument("--plan-file", help="grade an existing plan from this file")
    args = parser.parse_args()

    workflow = args.workflow_opt or " ".join(args.workflow)
    if not workflow.strip():
        print("No workflow described. Nothing to evaluate.")
        sys.exit(1)

    existing_plan = None
    if args.plan_file:
        with open(args.plan_file, encoding="utf-8") as f:
            existing_plan = f.read()

    print("\nEvaluating...\n")
    try:
        result = evaluate(workflow, plan=existing_plan)
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)

    print(format_report(result))
