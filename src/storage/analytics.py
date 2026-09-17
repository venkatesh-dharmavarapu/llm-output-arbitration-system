import json
from typing import Dict, Any
from src.storage.db import get_db_connection

def calculate_critic_analytics() -> Dict[str, Any]:
    """Aggregates performance and behavioral patterns across all historical evaluations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM arbitrations")
    rows = cursor.fetchall()
    conn.close()

    total_runs = len(rows)
    if total_runs == 0:
        return {
            "total_arbitrations": 0,
            "message": "No historical arbitration records logged yet."
        }

    critic_issues_found = {"factual_accuracy": 0, "logical_consistency": 0, "completeness": 0}
    critic_overrule_counts = {"factual_accuracy": 0, "logical_consistency": 0, "completeness": 0}
    total_scores = []
    total_disagreements = 0

    for row in rows:
        total_scores.append(row["overall_quality_score"])
        disagreements = json.loads(row["disagreements_json"])
        total_disagreements += len(disagreements)
        
        reports = json.loads(row["critic_reports_json"])
        for r in reports:
            dim = r.get("dimension")
            if dim in critic_issues_found:
                critic_issues_found[dim] += len(r.get("issues", []))

        verdict = json.loads(row["verdict_json"])
        dismissed = verdict.get("dismissed_flags", [])
        for d in dismissed:
            dim = d.get("raised_by_dimension")
            if dim in critic_overrule_counts:
                critic_overrule_counts[dim] += 1

    avg_score = round(sum(total_scores) / total_runs, 2)
    
    # Most proactive critic (found most raw issues)
    most_proactive = max(critic_issues_found, key=critic_issues_found.get) if critic_issues_found else "N/A"
    
    # Critic overruled most often by Chief Adjudicator
    most_overruled = max(critic_overrule_counts, key=critic_overrule_counts.get) if critic_overrule_counts else "N/A"

    return {
        "total_arbitrations": total_runs,
        "average_quality_score": avg_score,
        "total_cross_model_disagreements": total_disagreements,
        "critic_issues_detected": critic_issues_found,
        "critic_overruled_by_adjudicator": critic_overrule_counts,
        "key_insights": {
            "most_vigilant_critic": most_proactive,
            "most_overruled_critic": most_overruled,
            "disagreement_rate_per_run": round(total_disagreements / total_runs, 2)
        }
    }