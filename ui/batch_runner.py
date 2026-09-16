import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Dict, Any
from src.graph.workflow import arbitration_graph

def run_batch_arbitration(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Runs arbitration sequentially across a list of prompt-response pairs."""
    results = []
    for item in items:
        state = arbitration_graph.invoke({
            "prompt": item["prompt"],
            "response_text": item["response_text"],
            "critic_reports": [],
            "disagreements": []
        })
        verdict = state.get("final_verdict") or {}
        confirmed = verdict.get("confirmed_issues", [])
        
        results.append({
            "Prompt Excerpt": item["prompt"][:50] + "...",
            "Response Excerpt": item["response_text"][:60] + "...",
            "Overall Score": f"{verdict.get('overall_quality_score', 0)}/10",
            "Confidence": f"{verdict.get('confidence_score', 0.0):.2f}",
            "Confirmed Issues": len(confirmed),
            "Executive Summary": verdict.get("executive_summary", "N/A")
        })
    return results