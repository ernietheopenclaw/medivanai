"""Clinical report generation via NIM/Ollama LLM for MediVan AI."""
import httpx
import logging
from datetime import datetime, timezone
from backend.config import MOCK_MODE, NIM_ENDPOINT, NIM_MODEL

logger = logging.getLogger(__name__)


def _call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str | None:
    """Call LLM via OpenAI-compatible API (NIM, Ollama, vLLM, etc.)."""
    try:
        resp = httpx.post(
            f"{NIM_ENDPOINT}/chat/completions",
            json={
                "model": NIM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a clinical decision support AI for MediVan AI, a mobile health screening platform. Generate professional, evidence-based clinical reports. Be specific, cite findings data, and always include appropriate disclaimers."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        logger.warning(f"LLM endpoint unreachable at {NIM_ENDPOINT}")
        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def generate_report(session: dict) -> str:
    """Generate a holistic patient screening report from all session findings."""
    if MOCK_MODE:
        return _mock_report(session)

    findings_text = _format_findings(session["findings"])
    prompt = f"""Generate a comprehensive patient screening report for a mobile health unit visit.

SESSION DATA:
- Session ID: {session['id']}
- Date: {session['created_at']}
- Total analyses: {len(session['findings'])}

FINDINGS:
{findings_text}

REPORT FORMAT (follow exactly):
═══════════════════════════════════════════════
  MEDIVAN AI — PATIENT SCREENING REPORT
  Mobile Health Unit | [Date] | Session: [ID]
═══════════════════════════════════════════════

SCREENING SUMMARY
  - Modalities analyzed and count
  - Overall triage level (LOW/MODERATE/HIGH/URGENT)

FINDINGS (for each modality):
  [Number]. [SPECIALTY] — [Body Part/Image Type]
    Classification: [result]
    Confidence: [X]%
    Risk Level: [emoji + level]
    Recommendation: [specific clinical action]

HOLISTIC ASSESSMENT
  - Cross-modality pattern synthesis
  - Systemic disease indicators (e.g., DR + hypertension suggesting metabolic syndrome)
  - Notable correlations between findings

PRIORITY REFERRALS
  - Numbered list ordered by urgency
  - Include timeframe for each referral

DISCLAIMER
  - AI screening tool, not a diagnosis
  - Physician confirmation required

Use medical terminology appropriately. Be specific about the actual findings — don't generate generic text."""

    result = _call_llm(prompt)
    if result:
        return result

    # Fallback: generate report from template
    logger.warning("LLM unavailable, generating template report")
    return _template_report(session)


def generate_explanation(image_type: str, result: dict) -> str:
    """Generate a plain-English explanation for a single finding."""
    if MOCK_MODE:
        return result.get("recommendation", "")

    classification = result.get("classification", "unknown")
    confidence = result.get("confidence", 0)
    risk = result.get("risk_level", "unknown")

    prompt = f"""In 2-3 clinical sentences, explain this {image_type.replace('_', ' ')} finding for a clinician:
- Classification: {classification}
- Confidence: {confidence*100:.1f}%
- Risk Level: {risk}
Include what this means clinically and immediate next steps."""

    result_text = _call_llm(prompt, max_tokens=200)
    if result_text:
        return result_text
    return result.get("recommendation", f"{classification} detected with {confidence*100:.1f}% confidence. Risk level: {risk}.")


def _format_findings(findings: list) -> str:
    parts = []
    for i, f in enumerate(findings, 1):
        line = f"[{i}] Type: {f.get('image_type', 'unknown')}"
        line += f" | Classification: {f.get('classification', 'N/A')}"
        line += f" | Confidence: {f.get('confidence', 0)*100:.1f}%"
        line += f" | Risk: {f.get('risk_level', 'unknown')}"
        if 'recommendation' in f:
            line += f" | Rec: {f['recommendation']}"
        if 'dr_grade' in f:
            line += f" | DR Grade: {f['dr_grade']}"
        if 'severity_score' in f:
            line += f" | Severity: {f['severity_score']}"
        parts.append(line)
    return "\n".join(parts) if parts else "No findings recorded."


def _template_report(session: dict) -> str:
    """Generate a structured report without LLM (template-based fallback)."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
    findings = session.get("findings", [])
    modalities = list(set(f.get("image_type", "unknown") for f in findings))

    risk_levels = [f.get("risk_level", "low") for f in findings]
    if "high" in risk_levels:
        triage = "⚠️ HIGH — Immediate referral recommended"
    elif "moderate" in risk_levels:
        triage = "🟡 MODERATE — Follow-up within 2 weeks"
    else:
        triage = "🟢 LOW — Routine follow-up"

    findings_section = ""
    for i, f in enumerate(findings, 1):
        risk_emoji = {"high": "🔴", "moderate": "🟡", "low": "🟢"}.get(f.get("risk_level", "low"), "⚪")
        findings_section += f"""
  {i}. {f.get('image_type', 'UNKNOWN').upper().replace('_', ' ')}
    Classification: {f.get('classification', 'N/A')}
    Confidence: {f.get('confidence', 0)*100:.1f}%
    Risk Level: {risk_emoji} {f.get('risk_level', 'unknown').upper()}
    Recommendation: {f.get('recommendation', 'Consult specialist')}
"""

    return f"""═══════════════════════════════════════════════
  MEDIVAN AI — PATIENT SCREENING REPORT
  Mobile Health Unit | {now}
  Session: {session['id']}
═══════════════════════════════════════════════

SCREENING SUMMARY
  Modalities: {', '.join(m.replace('_', ' ').title() for m in modalities)}
  Analyses performed: {len(findings)}
  Overall Triage: {triage}

FINDINGS
{findings_section if findings_section.strip() else '  No findings recorded.'}

HOLISTIC ASSESSMENT
  {_cross_modality_note(findings)}

PRIORITY REFERRALS
{_priority_referrals(findings)}

═══════════════════════════════════════════════
⚕️ DISCLAIMER: MediVan AI is an AI-assisted
screening tool for research/demonstration
purposes. This does NOT constitute a medical
diagnosis. All findings require confirmation
by a licensed physician. Clinical management
decisions must be based on comprehensive
evaluation by qualified healthcare providers.
═══════════════════════════════════════════════"""


def _mock_report(session: dict) -> str:
    """Generate a realistic mock report for demo purposes."""
    return _template_report(session)


def _cross_modality_note(findings: list) -> str:
    types = [f.get("image_type") for f in findings]
    notes = []

    # DR + anything = diabetes screening flag
    dr_findings = [f for f in findings if f.get("image_type") == "fundus"]
    for f in dr_findings:
        if f.get("risk_level") in ("moderate", "high"):
            notes.append(f"Retinopathy finding ({f.get('classification', 'unknown')}) suggests possible inadequate glycemic control. Recommend HbA1c testing and endocrinology consultation.")

    # Skin cancer + chest = systemic screening note
    skin_findings = [f for f in findings if f.get("image_type") == "skin_lesion"]
    for f in skin_findings:
        if f.get("classification") in ("melanoma", "basal cell carcinoma"):
            notes.append(f"Suspicious skin lesion ({f.get('classification')}) identified. If melanoma confirmed, staging workup including chest imaging recommended.")

    # Chest + DR = cardiovascular risk
    if "chest_xray" in types and "fundus" in types:
        chest_findings = [f for f in findings if f.get("image_type") == "chest_xray"]
        has_cardiac = any(f.get("classification") == "cardiomegaly" for f in chest_findings)
        has_dr = any(f.get("risk_level") in ("moderate", "high") for f in dr_findings)
        if has_cardiac and has_dr:
            notes.append("Concurrent cardiomegaly and diabetic retinopathy suggest significant cardiovascular risk. Comprehensive metabolic and cardiac workup recommended.")

    if len(types) > 1 and not notes:
        notes.append("Multi-system screening completed. No concerning cross-modality patterns identified at this time.")
    elif not notes:
        notes.append("Single modality screening. Consider additional modalities for comprehensive assessment.")

    return " ".join(notes)


def _priority_referrals(findings: list) -> str:
    high = [f for f in findings if f.get("risk_level") == "high"]
    mod = [f for f in findings if f.get("risk_level") == "moderate"]
    lines = []
    for i, f in enumerate(high, 1):
        specialty = {
            "skin_lesion": "Dermatology", "chest_xray": "Pulmonology/Radiology", "fundus": "Ophthalmology"
        }.get(f.get("image_type"), "Specialist")
        lines.append(f"  {i}. [URGENT] {specialty}: {f.get('classification', 'abnormal finding')} — Immediate referral")
    for i, f in enumerate(mod, len(high) + 1):
        specialty = {
            "skin_lesion": "Dermatology", "chest_xray": "Cardiology/Pulmonology", "fundus": "Ophthalmology"
        }.get(f.get("image_type"), "Specialist")
        lines.append(f"  {i}. [ROUTINE] {specialty}: {f.get('classification', 'finding')} — Follow-up within 2-4 weeks")
    return "\n".join(lines) if lines else "  No urgent referrals indicated. Routine follow-up recommended."
