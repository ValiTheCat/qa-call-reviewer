"""QA Call Reviewer - analyzer v2.1.

Reads a call (account context + transcript), sends it to Claude,
and returns a scorecard. Compliance rules are enforced in code.

v2 changes:
- Disclosure must reach the decision maker (new "needs_review" result)
- Clear definition of a strong budget pitch
- Compliance rules enforced in code, not left to the AI
v2.1 changes:
- Notes when a code rule escalated the status ("rule_applied")
- API key loads from the local secrets file OR Streamlit Cloud secrets
v2.2 changes:
- Validates the AI's JSON and fills any missing fields with safe defaults,
  flagging the call for human review instead of crashing
v3 changes:
- Structured output: the AI must fill in a defined schema (via a forced tool
  call) instead of being asked nicely to write JSON
- Validation also catches values outside the allowed options
v4 changes:
- Handles audio transcripts: speakers arrive as "Speaker A/B/C" with [m:ss]
  timestamps; the AI identifies who is the rep, gatekeeper and advertiser
- Each finding includes the timestamp so auditors can jump to the moment
"""
import json
import tomllib
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
PROMPT_VERSION = "v4"

SYSTEM_PROMPT = """You are a QA auditor reviewing sales calls for an advertising platform.
Review the call transcript using the account context provided, and check:

1. Recorded-line disclosure: The rep must tell the decision maker (the advertiser)
   that the call is recorded for quality purposes, near the start of the conversation.
   - pass: the advertiser heard the disclosure, OR no decision maker was reached and
     the disclosure was given before leaving a message with a gatekeeper.
   - needs_review: the disclosure was only given to a gatekeeper or assistant, and
     not repeated after the advertiser joined the call.
   - fail: no disclosure was given at all.

2. Budget pitch: A strong pitch has all three of: (a) a reference to current spend
   or performance, (b) a specific proposed amount or change (an increase OR a
   reallocation), and (c) an expected outcome tied to the advertiser's goal.
   Missing any of these is weak. Mentioning the word "budget" is not enough.
   If no decision maker was reached, mark not_applicable.

3. Objections: What objections did the advertiser raise, how did the rep respond,
   and was it overcome?

4. Coupon: Only applies if the account context shows an active coupon AND the rep
   reached the decision maker. Otherwise mark not_applicable.

AUDIO TRANSCRIPTS: Some transcripts come from audio recordings. Speakers are
labeled "Speaker A", "Speaker B", and so on, and each line starts with an [m:ss]
timestamp. First work out which speaker is the rep, the advertiser (decision
maker), and any gatekeeper from what they say, then apply the checks above.
Transcription can contain small errors; judge meaning, not exact wording.
When a transcript has timestamps, give the m:ss of each piece of evidence.

Submit your review using the submit_scorecard tool."""

ALLOWED = {
    "disclosure": ["pass", "needs_review", "fail"],
    "budget_pitch": ["strong", "weak", "not_applicable"],
    "coupon": ["mentioned", "missed", "not_applicable"],
    "outcome": ["overcome", "partial", "not_overcome"],
    "status": ["pass", "coaching", "review", "high_priority"],
}


TIMESTAMP = {"type": "string",
             "description": "m:ss where the evidence occurs, if the transcript has timestamps; otherwise empty."}


def _check(allowed: list, with_evidence: bool = True) -> dict:
    props = {"result": {"type": "string", "enum": allowed},
             "reason": {"type": "string", "description": "One sentence."}}
    if with_evidence:
        props["evidence"] = {"type": "string", "description": "Short quote from the transcript."}
    required = list(props)
    if with_evidence:
        props["timestamp"] = TIMESTAMP
    return {"type": "object", "properties": props, "required": required}


# The schema the AI must fill in. Forcing this tool guarantees the structure.
SCORECARD_TOOL = {
    "name": "submit_scorecard",
    "description": "Submit the QA scorecard for this call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "disclosure": _check(ALLOWED["disclosure"]),
            "budget_pitch": _check(ALLOWED["budget_pitch"]),
            "objections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "objection": {"type": "string"},
                        "rep_response": {"type": "string"},
                        "outcome": {"type": "string", "enum": ALLOWED["outcome"]},
                        "timestamp": TIMESTAMP,
                    },
                    "required": ["objection", "rep_response", "outcome"],
                },
            },
            "coupon": _check(ALLOWED["coupon"], with_evidence=False),
            "speaker_roles": {
                "type": "string",
                "description": "For audio transcripts, who each speaker is, e.g. "
                               "'A = Rep, B = Gatekeeper, C = Advertiser'. Empty otherwise.",
            },
            "overall": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ALLOWED["status"]},
                    "summary": {"type": "string", "description": "One or two sentences."},
                },
                "required": ["status", "summary"],
            },
        },
        "required": ["disclosure", "budget_pitch", "objections", "coupon", "overall"],
    },
}

# Severity order, lowest to highest
SEVERITY = ["pass", "coaching", "review", "high_priority"]

_client = None


def get_secret(name: str) -> str:
    """Local runs read .streamlit/secrets.toml; Streamlit Cloud uses st.secrets."""
    path = Path(".streamlit/secrets.toml")
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)[name]
    import streamlit as st
    return st.secrets[name]


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    return _client


def escalate(current: str, minimum: str) -> str:
    """Raise the status to at least `minimum`, never lower it."""
    if current not in SEVERITY:
        return minimum
    return minimum if SEVERITY.index(minimum) > SEVERITY.index(current) else current


def normalize(result: dict) -> dict:
    """Make sure every field the app needs exists.

    The AI sometimes skips or renames a field. Instead of crashing, fill in a
    safe default and send the call to a human by marking it for review.
    """
    problems = []

    for check in ("disclosure", "budget_pitch", "coupon"):
        part = result.get(check)
        if not isinstance(part, dict):
            part = {}
            problems.append(check)
        if part.get("result") not in ALLOWED[check]:
            part["result"] = "needs_review" if check == "disclosure" else "not_applicable"
            problems.append(f"{check} result")
        part.setdefault("evidence", "")
        part.setdefault("reason", "")
        part.setdefault("timestamp", "")
        result[check] = part

    if not isinstance(result.get("objections"), list):
        result["objections"] = []
        problems.append("objections")
    for o in result["objections"]:
        o.setdefault("objection", "Not specified")
        o.setdefault("rep_response", "Not specified")
        o.setdefault("timestamp", "")
        if o.get("outcome") not in ALLOWED["outcome"]:
            o["outcome"] = "partial"
            problems.append("objection outcome")

    if not isinstance(result.get("speaker_roles"), str):
        result["speaker_roles"] = ""

    overall = result.get("overall")
    if not isinstance(overall, dict):
        overall = {}
        problems.append("overall")
    if overall.get("status") not in ALLOWED["status"]:
        overall["status"] = "review"
        problems.append("overall status")
    if not overall.get("summary"):
        overall["summary"] = "The AI did not return a summary for this call. Review it manually."
        problems.append("summary")
    result["overall"] = overall

    if problems:
        overall["status"] = escalate(overall["status"], "review")
        overall["format_warning"] = "Missing from AI response: " + ", ".join(problems)
    return result


def apply_business_rules(result: dict) -> dict:
    """Compliance rules enforced in code so the AI can't soften them."""
    original = result["overall"]["status"]
    disclosure = result["disclosure"]["result"]
    status = original

    if disclosure == "fail":
        status = escalate(status, "high_priority")
    elif disclosure == "needs_review":
        status = escalate(status, "review")

    result["overall"]["status"] = status
    if status != original:
        result["overall"]["rule_applied"] = (
            f"Escalated from {original} to {status} by compliance rule "
            f"(disclosure: {disclosure})"
        )
    return result


def analyze_call(call_text: str) -> dict:
    """Send one call to Claude and return the scorecard as a dict."""
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": call_text}],
        tools=[SCORECARD_TOOL],
        tool_choice={"type": "tool", "name": "submit_scorecard"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return apply_business_rules(normalize(dict(block.input)))


if __name__ == "__main__":
    for call_file in sorted(Path("calls").glob("*.txt")):
        print(f"\n===== {call_file.name} ({PROMPT_VERSION}) =====")
        try:
            result = analyze_call(call_file.read_text(encoding="utf-8"))
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
