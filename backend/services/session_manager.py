"""Patient screening session management."""
import uuid
from datetime import datetime, timezone
from typing import Optional

_sessions: dict = {}


def create_session() -> dict:
    sid = str(uuid.uuid4())[:8]
    session = {
        "id": sid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "findings": [],
        "report": None,
    }
    _sessions[sid] = session
    return session


def get_session(sid: str) -> Optional[dict]:
    return _sessions.get(sid)


def add_finding(sid: str, finding: dict) -> dict:
    session = _sessions.get(sid)
    if not session:
        raise ValueError(f"Session {sid} not found")
    finding["timestamp"] = datetime.now(timezone.utc).isoformat()
    finding["index"] = len(session["findings"])
    session["findings"].append(finding)
    return finding


def set_report(sid: str, report: str):
    session = _sessions.get(sid)
    if not session:
        raise ValueError(f"Session {sid} not found")
    session["report"] = report


def list_sessions() -> list:
    return list(_sessions.values())
