import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

DB_PATH = "arbitration.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes audit trail tables for arbitrations and critic evaluations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS arbitrations (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response_text TEXT NOT NULL,
        overall_quality_score INTEGER NOT NULL,
        confidence_score REAL NOT NULL,
        executive_summary TEXT,
        confirmed_issues_count INTEGER DEFAULT 0,
        dismissed_flags_count INTEGER DEFAULT 0,
        verdict_json TEXT NOT NULL,
        critic_reports_json TEXT NOT NULL,
        disagreements_json TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def save_arbitration(
    prompt: str,
    response_text: str,
    verdict: Dict[str, Any],
    critic_reports: List[Any],
    disagreements: List[Any]
) -> str:
    """Saves a completed arbitration run into the audit trail and returns its unique ID."""
    record_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    
    # Extract report and disagreement dictionaries safely
    reports_data = [r.model_dump() if hasattr(r, "model_dump") else r for r in critic_reports]
    disagreements_data = [d.model_dump() if hasattr(d, "model_dump") else d for d in disagreements]
    
    confirmed = verdict.get("confirmed_issues", [])
    dismissed = verdict.get("dismissed_flags", [])
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO arbitrations (
        id, created_at, prompt, response_text, overall_quality_score,
        confidence_score, executive_summary, confirmed_issues_count,
        dismissed_flags_count, verdict_json, critic_reports_json, disagreements_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_id,
        created_at,
        prompt,
        response_text,
        verdict.get("overall_quality_score", 0),
        verdict.get("confidence_score", 0.0),
        verdict.get("executive_summary", ""),
        len(confirmed),
        len(dismissed),
        json.dumps(verdict),
        json.dumps(reports_data),
        json.dumps(disagreements_data)
    ))
    
    conn.commit()
    conn.close()
    return record_id

def get_arbitration_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a past arbitration record from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM arbitrations WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "prompt": row["prompt"],
        "response_text": row["response_text"],
        "overall_quality_score": row["overall_quality_score"],
        "confidence_score": row["confidence_score"],
        "executive_summary": row["executive_summary"],
        "confirmed_issues_count": row["confirmed_issues_count"],
        "dismissed_flags_count": row["dismissed_flags_count"],
        "verdict": json.loads(row["verdict_json"]),
        "critic_reports": json.loads(row["critic_reports_json"]),
        "disagreements": json.loads(row["disagreements_json"])
    }