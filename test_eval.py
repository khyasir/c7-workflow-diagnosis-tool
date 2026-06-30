"""Tests for eval.py.

Two layers:
  - Offline tests (no API key): clamp logic + report formatting with a fake judge.
  - Live runner: actually calls the LLM on every workflow in test_inputs.json,
    but only when GROQ_API_KEY is set (skips otherwise).

Run offline tests:   python test_eval.py
Run live on inputs:  python test_eval.py --live
"""

import json
import os
import sys

import eval as ev


def _fake_judge_result():
    """A judge payload with deliberately out-of-band scores to test clamping."""
    scores = [
        {"principle": name, "score": 10.0, "comment": f"ok {name}"}  # 10 > 6.67 band
        for name, _ in ev.PRINCIPLES
    ]
    scores[0]["score"] = -5.0  # below 0 → should clamp to 0
    return {
        "scores": scores,
        "top_3_improvements": ["be more specific", "name tools", "quantify payoff"],
        "suggested_edits": ["add a time-saved estimate"],
        "rewrite": "REWRITTEN PLAN",
    }


def test_clamp_and_overall(monkeypatch_judge=None):
    """Scores clamp to [0, 6.67] per item and overall caps at 100."""
    ev._judge = lambda wf, plan: _fake_judge_result()  # stub the LLM
    result = ev.evaluate("some workflow", plan="some plan")

    assert result["scores"][0]["score"] == 0.0, "negative score must clamp to 0"
    assert all(s["score"] <= ev.POINTS_EACH for s in result["scores"]), "score over band"
    assert result["overall"] <= 100.0, "overall must cap at 100"
    # 14 items clamped to 6.67 -> ~93.4, well under 100 even though raw was 140.
    print(f"  clamp/overall ok — overall={result['overall']}")


def test_format_report():
    """Report contains the header, a row per principle, and the rewrite."""
    ev._judge = lambda wf, plan: _fake_judge_result()
    result = ev.evaluate("some workflow", plan="some plan")
    report = ev.format_report(result)

    assert "Overall Score:" in report
    assert "Score Breakdown:" in report
    assert report.count("\n| ") >= len(ev.PRINCIPLES), "missing principle rows"
    assert "REWRITTEN PLAN" in report
    assert "Top 3 Areas to Improve:" in report
    print("  format_report ok")


def run_offline():
    print("Offline tests:")
    test_clamp_and_overall()
    test_format_report()
    print("All offline tests passed.\n")


def run_live():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — skipping live run.")
        return
    # Re-import to undo any stubbing of _judge from offline tests.
    import importlib
    importlib.reload(ev)

    with open("test_inputs.json", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        print(f"\n{'='*70}\nCASE: {case['name']}\n{'='*70}")
        result = ev.evaluate(case["workflow"])
        print(ev.format_report(result))


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live()
    else:
        run_offline()
