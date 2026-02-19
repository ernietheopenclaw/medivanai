"""NIM LLM report generation."""
import httpx
from datetime import datetime, timezone
from backend.config import MOCK_MODE, NIM_ENDPOINT, NIM_MODEL


def generate_report(session: dict) -> str:
    if MOCK_MODE:
        return _mock_report(session)

    findings_text = _format_findings(session["findings"])
    prompt = f"""You are a clinical decision support system for a mobile health screening unit (MediVan AI).
Generate a structured patient screening report based on these findings.

Session ID: {session['id']}
Date: {session['created_at']}
Number of analyses: {len(session['findings'])}

FINDINGS:
{findings_text}

Generate the report in this exact structure:
1. SCREENING SUMMARY - modalities analyzed, overall triage level
2. FINDINGS - per modality with classification, confidence, risk level, recommendation
3. HOLISTIC ASSESSMENT - cross-modality synthesis, pattern recognition
4. PRIORITY REFERRALS - ordered by urgency
5. DISCLAIMER - not a diagnosis, requires physician confirmation

Be specific and clinical. Use the actual findings data."""

    try:
        resp = httpx.post(
            f"{NIM_ENDPOINT}/chat/completions",
            json={
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Report generation failed ({e}). Findings summary:\n{findings_text}"


def generate_explanation(image_type: str, result: dict) -> str:
    if MOCK_MODE:
        return result.get("recommendation", "")
    try:
        prompt = f"Explain this {image_type} finding in plain clinical English (2-3 sentences): {result['classification']} with {result['confidence']*100:.1f}% confidence. Risk: {result['risk_level']}."
        resp = httpx.post(
            f"{NIM_ENDPOINT}/chat/completions",
            json={"model": NIM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.3},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except:
        return result.get("recommendation", "")


def _format_findings(findings: list) -> str:
    parts = []
    for i, f in enumerate(findings):
        parts.append(f"[{i+1}] Type: {f['image_type']} | Classification: {f['classification']} | Confidence: {f['confidence']} | Risk: {f['risk_level']}")
    return "\n".join(parts) if parts else "No findings recorded."


def _mock_report(session: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
    findings = session.get("findings", [])
    modalities = list(set(f["image_type"] for f in findings)) if findings else ["none"]

    risk_levels = [f.get("risk_level", "low") for f in findings]
    if "high" in risk_levels:
        triage = "HIGH — Immediate referral recommended"
    elif "moderate" in risk_levels:
        triage = "MODERATE — Follow-up within 2 weeks"
    else:
        triage = "LOW — Routine follow-up"

    findings_section = ""
    for f in findings:
        findings_section += f"""
  • {f['image_type'].upper().replace('_', ' ')}
    Classification: {f['classification']}
    Confidence: {f['confidence']*100:.1f}%
    Risk Level: {f['risk_level'].upper()}
    Recommendation: {f.get('recommendation', 'See specialist')}
"""

    return f"""═══════════════════════════════════════════════
  MEDIVAN AI — PATIENT SCREENING REPORT
  Mobile Health Unit | {now}
  Session ID: {session['id']}
═══════════════════════════════════════════════

SCREENING SUMMARY
  Modalities analyzed: {', '.join(m.replace('_', ' ').title() for m in modalities)}
  Total analyses: {len(findings)}
  Overall triage level: {triage}

FINDINGS
{findings_section if findings_section else '  No findings recorded.'}

HOLISTIC ASSESSMENT
  {'Multiple modality screening completed. ' if len(modalities) > 1 else 'Single modality screening. '}
  {_cross_modality_note(findings)}

PRIORITY REFERRALS
{_priority_referrals(findings)}

═══════════════════════════════════════════════
DISCLAIMER: This is an AI-assisted screening
tool and does NOT constitute a medical diagnosis.
All findings must be confirmed by a qualified
physician. Patient management decisions should
be based on comprehensive clinical evaluation.
═══════════════════════════════════════════════"""


def _cross_modality_note(findings: list) -> str:
    types = [f["image_type"] for f in findings]
    notes = []
    if "fundus" in types and any(f["risk_level"] in ("moderate", "high") for f in findings if f["image_type"] == "fundus"):
        notes.append("Retinopathy findings suggest possible uncontrolled diabetes. Recommend HbA1c testing and endocrinology consultation.")
    if "chest_xray" in types and "skin_lesion" in types:
        notes.append("Multi-system screening completed. Cross-reference findings for systemic conditions.")
    if not notes:
        notes.append("No cross-modality patterns identified requiring additional workup at this time.")
    return " ".join(notes)


def _priority_referrals(findings: list) -> str:
    high = [f for f in findings if f.get("risk_level") == "high"]
    mod = [f for f in findings if f.get("risk_level") == "moderate"]
    lines = []
    for i, f in enumerate(high, 1):
        lines.append(f"  {i}. [URGENT] {f['image_type'].replace('_',' ').title()}: {f['classification']} — Immediate specialist referral")
    for i, f in enumerate(mod, len(high) + 1):
        lines.append(f"  {i}. [ROUTINE] {f['image_type'].replace('_',' ').title()}: {f['classification']} — Follow-up within 2-4 weeks")
    return "\n".join(lines) if lines else "  No urgent referrals indicated."

