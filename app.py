"""QA Call Reviewer - Streamlit app.

Auditors load calls, the AI gives every call a first pass, and the
priority queue shows which calls need a human to listen first.
"""
import csv
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import SEVERITY, analyze_call

st.set_page_config(page_title="QA Call Reviewer", page_icon="🎧", layout="wide")

# ---------- Look and feel ----------
STATUS = {
    "high_priority": ("High priority", "#B3261E"),
    "review": ("Needs review", "#B26A00"),
    "coaching": ("Coaching", "#2F5D9F"),
    "pass": ("Pass", "#2E7D4F"),
}
CHECK_LABELS = {
    "pass": "Pass", "fail": "Fail", "needs_review": "Needs review",
    "strong": "Strong", "weak": "Weak", "not_applicable": "Not applicable",
    "mentioned": "Mentioned", "missed": "Missed",
    "overcome": "Overcome", "partial": "Partial", "not_overcome": "Not overcome",
}
GOOD = {"pass", "strong", "mentioned", "overcome"}
BAD = {"fail", "weak", "missed", "not_overcome"}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block-container { max-width: 1150px; padding-top: 2.5rem; }
.stats { display: flex; gap: 2.5rem; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.stat .n { font-size: 1.9rem; font-weight: 600; line-height: 1.1; }
.stat .l { font-size: 0.85rem; color: #5A6478; }
.queue-row { display: grid; grid-template-columns: 14rem 1fr; gap: 1.2rem;
             border-left: 6px solid var(--c); background: #FFFFFF; padding: 0.9rem 1.1rem;
             margin-bottom: 0.6rem; border-radius: 4px; }
.queue-row .who { font-weight: 600; font-size: 1.05rem; }
.queue-row .tag { color: var(--c); font-weight: 600; font-size: 0.95rem; margin-top: 0.15rem; }
.queue-row .why { max-width: 75ch; }
.queue-row .rule { margin-top: 0.35rem; font-size: 0.82rem; color: #5A6478; }
.badges { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.55rem; }
.badge { font-size: 0.8rem; padding: 0.15rem 0.55rem; border-radius: 999px; border: 1px solid currentColor; }
.badge.good { color: #2E7D4F; background: #EEF6F1; }
.badge.bad { color: #B3261E; background: #FBEDEC; }
.badge.mid { color: #B26A00; background: #FBF3E6; }
.badge.na { color: #8A93A6; background: #F3F4F7; }
@media (max-width: 700px) { .queue-row { grid-template-columns: 1fr; gap: 0.4rem; } }
.check { background: #FFFFFF; padding: 0.7rem 0.9rem; border-radius: 4px; height: 100%; }
.check .name { font-size: 0.85rem; color: #5A6478; }
.check .res { font-weight: 600; font-size: 1.05rem; }
.good { color: #2E7D4F; } .bad { color: #B3261E; } .mid { color: #B26A00; } .na { color: #8A93A6; }
.quote { font-style: italic; margin-top: 0.3rem; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def review_call(text: str) -> dict:
    """Cached so the same call is never paid for twice."""
    return analyze_call(text)


@st.cache_data(show_spinner=False)
def transcribe(audio_bytes: bytes) -> str:
    """Cached so the same recording is never transcribed twice."""
    from transcriber import transcribe_audio  # only needed for audio
    return transcribe_audio(audio_bytes)


def load_accounts() -> dict:
    with open("accounts.csv", newline="", encoding="utf-8") as f:
        return {row["account"]: row for row in csv.DictReader(f)}


def context_line(acct: dict) -> str:
    return (f"ACCOUNT CONTEXT: {acct['account']} | Monthly spend: {acct['monthly_spend']} | "
            f"Active coupon: {acct['coupon']} | Goal: {acct['goal']}")


def stamp(item: dict) -> str:
    return (item.get("timestamp") or "").strip()


def account_name(text: str, fallback: str) -> str:
    """Pull the account name from the ACCOUNT CONTEXT line."""
    for line in text.splitlines():
        if line.upper().startswith("ACCOUNT CONTEXT:"):
            return line.split(":", 1)[1].split("|")[0].strip() or fallback
    return fallback


def result_class(value: str) -> str:
    if value in GOOD:
        return "good"
    if value in BAD:
        return "bad"
    if value in ("needs_review", "partial"):
        return "mid"
    return "na"


def objection_result(objections: list) -> str:
    outcomes = [o["outcome"] for o in objections]
    if not outcomes:
        return "not_applicable"
    for worst in ("not_overcome", "partial"):
        if worst in outcomes:
            return worst
    return "overcome"


def badges(r: dict) -> str:
    items = [
        ("Disclosure", r["disclosure"]["result"]),
        ("Budget pitch", r["budget_pitch"]["result"]),
        ("Coupon", r["coupon"]["result"]),
        ("Objections", objection_result(r["objections"])),
    ]
    spans = []
    for name, value in items:
        label = "None" if (name == "Objections" and value == "not_applicable") else CHECK_LABELS.get(value, value)
        spans.append(f'<span class="badge {result_class(value)}">{name}: {label}</span>')
    return '<div class="badges">' + "".join(spans) + "</div>"


def check_card(title: str, value: str, evidence: str = "", ts: str = "") -> str:
    label = CHECK_LABELS.get(value, value)
    at = f"{html.escape(ts)} " if ts else ""
    quote = f'<div class="quote">{at}"{html.escape(evidence)}"</div>' if evidence and value != "not_applicable" else ""
    return (f'<div class="check"><div class="name">{title}</div>'
            f'<div class="res {result_class(value)}">{label}</div>{quote}</div>')


# ---------- State ----------
st.session_state.setdefault("calls", {})     # name -> transcript text
st.session_state.setdefault("results", {})   # name -> scorecard
st.session_state.setdefault("audio", {})     # name -> recording bytes

# ---------- Sidebar: load and review calls ----------
with st.sidebar:
    st.header("Load calls")
    source = st.radio("Calls to review", ["Sample calls", "Upload transcripts", "Upload audio"])

    calls, audio_items = {}, []
    if source == "Sample calls":
        for f in sorted(Path("calls").glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            calls[account_name(text, f.stem)] = text
        st.caption(f"{len(calls)} mock calls with fictional accounts.")
    elif source == "Upload transcripts":
        uploads = st.file_uploader("Transcript files (.txt)", type="txt", accept_multiple_files=True)
        for up in uploads or []:
            text = up.read().decode("utf-8")
            calls[account_name(text, Path(up.name).stem)] = text
        st.caption("Start each file with an ACCOUNT CONTEXT line, then the transcript.")
    else:
        accounts = load_accounts()
        uploads = st.file_uploader("Call recordings", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        for up in uploads or []:
            acct = st.selectbox(f"Account for {up.name}", list(accounts), key=f"acct_{up.name}")
            audio_items.append((f"{acct} ({Path(up.name).stem})", up.getvalue(), accounts[acct]))
        st.caption("Recordings are transcribed with speaker labels and timestamps, then reviewed.")

    ready = bool(calls or audio_items)
    if st.button("Review calls", type="primary", disabled=not ready, use_container_width=True):
        st.session_state.results, st.session_state.audio = {}, {}
        progress = st.progress(0.0, text="Starting")
        for name, data, acct in audio_items:
            progress.progress(0.0, text=f"Transcribing {name}")
            try:
                transcript = transcribe(data)
                calls[name] = f"{context_line(acct)}\n\nTRANSCRIPT (from audio):\n{transcript}"
                st.session_state.audio[name] = data
            except Exception as e:
                st.error(f"{name}: could not be transcribed ({e})")
        st.session_state.calls = calls
        for i, (name, text) in enumerate(calls.items(), start=1):
            try:
                st.session_state.results[name] = review_call(text)
            except Exception as e:
                st.error(f"{name}: could not be reviewed ({e})")
            progress.progress(i / len(calls), text=f"Reviewed {i} of {len(calls)}")
        progress.empty()

# ---------- Main ----------
st.title("QA Call Reviewer")
st.write("Every call gets a first pass. Auditors start with the calls that need a human.")

results = st.session_state.results
if not results:
    st.info("Load calls in the sidebar, then select Review calls.")
    st.stop()

# Queue order: most severe first
queue = sorted(results, key=lambda n: SEVERITY.index(results[n]["overall"]["status"]), reverse=True)

# Auditor verdicts live in session state (keys created by the radio widgets)
verdicts = [st.session_state.get(f"verdict_{n}") for n in results]
agree, disagree = verdicts.count("Agree"), verdicts.count("Disagree")

high = sum(r["overall"]["status"] == "high_priority" for r in results.values())
review = sum(r["overall"]["status"] == "review" for r in results.values())
agreement = f"{agree / (agree + disagree):.0%}" if agree + disagree else "None yet"
stats = [(len(results), "calls reviewed"), (high, "high priority"),
         (review, "need review"), (agreement, "auditor agreement")]
st.markdown(
    '<div class="stats">' + "".join(
        f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in stats
    ) + "</div>",
    unsafe_allow_html=True,
)

tab_queue, tab_cards, tab_trends = st.tabs(["Listen first", "Call scorecards", "Team trends"])

# ----- Priority queue -----
with tab_queue:
    st.subheader("Listen to these first")
    for name in queue:
        overall = results[name]["overall"]
        label, color = STATUS[overall["status"]]
        rule = overall.get("rule_applied")
        rule_html = f'<div class="rule">Rule applied: {html.escape(rule)}</div>' if rule else ""
        warning = overall.get("format_warning")
        if warning:
            rule_html += f'<div class="rule">{html.escape(warning)}</div>'
        st.markdown(
            f'<div class="queue-row" style="--c:{color}">'
            f'<div><div class="who">{html.escape(name)}</div><div class="tag">{label}</div></div>'
            f'<div><div class="why">{html.escape(overall["summary"])}</div>'
            f'{badges(results[name])}{rule_html}</div></div>',
            unsafe_allow_html=True,
        )

# ----- Scorecards with auditor verdict -----
with tab_cards:
    name = st.selectbox("Call", queue)
    r = results[name]
    label, color = STATUS[r["overall"]["status"]]
    st.markdown(f'<h3 style="color:{color};margin-bottom:0">{label}</h3>', unsafe_allow_html=True)
    st.write(r["overall"]["summary"])
    if r["overall"].get("rule_applied"):
        st.caption(f'Rule applied: {r["overall"]["rule_applied"]}')
    if r["overall"].get("format_warning"):
        st.warning(r["overall"]["format_warning"])

    if name in st.session_state.audio:
        jumps = [("Disclosure", stamp(r["disclosure"])), ("Budget pitch", stamp(r["budget_pitch"]))]
        jumps += [(f"Objection {i}", stamp(o)) for i, o in enumerate(r["objections"], start=1)]
        jumps = [(label, ts) for label, ts in jumps if ts]
        if jumps:
            cols = st.columns(len(jumps) + 1)
            cols[0].markdown("**Jump to:**")
            for col, (label, ts) in zip(cols[1:], jumps):
                if col.button(f"{label} {ts}", key=f"jump_{name}_{label}"):
                    from transcriber import to_seconds
                    st.session_state[f"start_{name}"] = to_seconds(ts)
        st.audio(st.session_state.audio[name], start_time=st.session_state.get(f"start_{name}", 0))
        if r.get("speaker_roles"):
            st.caption(f"Speakers identified by AI: {r['speaker_roles']}")

    c1, c2, c3 = st.columns(3)
    c1.markdown(check_card("Recorded-line disclosure", r["disclosure"]["result"], r["disclosure"].get("evidence", ""), stamp(r["disclosure"])), unsafe_allow_html=True)
    c2.markdown(check_card("Budget pitch", r["budget_pitch"]["result"], r["budget_pitch"].get("evidence", ""), stamp(r["budget_pitch"])), unsafe_allow_html=True)
    c3.markdown(check_card("Coupon", r["coupon"]["result"]), unsafe_allow_html=True)

    st.markdown("#### Objections")
    if r["objections"]:
        for o in r["objections"]:
            outcome = CHECK_LABELS.get(o["outcome"], o["outcome"])
            at = f" ({stamp(o)})" if stamp(o) else ""
            st.markdown(f"**{o['objection']}**{at}  \nRep: {o['rep_response']}  \nOutcome: {outcome}")
    else:
        st.write("No objections raised.")

    st.markdown("#### Auditor verdict")
    st.radio("Do you agree with this review?", ["Not reviewed", "Agree", "Disagree"],
             key=f"verdict_{name}", horizontal=True)
    st.text_input("Note (optional)", key=f"note_{name}")

    with st.expander("Transcript"):
        st.text(st.session_state.calls.get(name, ""))

# ----- Team trends -----
with tab_trends:
    rows = []
    for n, r in results.items():
        rows.append({"Check": "Disclosure", "Result": CHECK_LABELS[r["disclosure"]["result"]]})
        rows.append({"Check": "Budget pitch", "Result": CHECK_LABELS[r["budget_pitch"]["result"]]})
        rows.append({"Check": "Coupon", "Result": CHECK_LABELS[r["coupon"]["result"]]})
        for o in r["objections"]:
            rows.append({"Check": "Objections", "Result": CHECK_LABELS.get(o["outcome"], o["outcome"])})
    df = pd.DataFrame(rows)
    counts = df.groupby(["Check", "Result"]).size().unstack(fill_value=0)
    st.subheader("Results by check")
    st.bar_chart(counts)

    st.subheader("Coaching themes")
    themes = {
        "Weak budget pitch": [n for n, r in results.items() if r["budget_pitch"]["result"] == "weak"],
        "Missed coupon": [n for n, r in results.items() if r["coupon"]["result"] == "missed"],
        "Objection not overcome": [n for n, r in results.items()
                                   if any(o["outcome"] == "not_overcome" for o in r["objections"])],
        "Disclosure issue": [n for n, r in results.items() if r["disclosure"]["result"] != "pass"],
    }
    for theme, names in themes.items():
        if names:
            st.write(f"**{theme}** ({len(names)}): {', '.join(names)}")
