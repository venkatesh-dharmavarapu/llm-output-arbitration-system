import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.storage.db import init_db, save_arbitration, get_arbitration_by_id
from src.storage.analytics import calculate_critic_analytics
from src.graph.workflow import arbitration_graph

# Initialize database schema on startup
init_db()

app = FastAPI(
    title="LLM Output Arbitration System API",
    description="Multi-agent evaluation API that audits candidate model outputs through parallel critics and adjudication.",
    version="1.0.0"
)

# --- Request / Response Contracts ---

class ArbitrationRequest(BaseModel):
    prompt: str = Field(..., example="What is the boiling point of water at sea level?")
    response_text: str = Field(..., example="Water boils at 100 degrees Celsius.")

class BatchArbitrationRequest(BaseModel):
    items: List[ArbitrationRequest]

class ArbitrationResponse(BaseModel):
    arbitration_id: str
    overall_quality_score: int
    confidence_score: float
    executive_summary: str
    confirmed_issues_count: int
    dismissed_flags_count: int
    verdict: dict
    critic_reports: list
    disagreements: list

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "arbitration-pipeline"}

@app.post("/v1/arbitrate", response_model=ArbitrationResponse, status_code=status.HTTP_201_CREATED)
def arbitrate_single(request: ArbitrationRequest):
    """Audits a single LLM output, logs the execution to SQLite, and returns the full verdict."""
    try:
        state = arbitration_graph.invoke({
            "prompt": request.prompt,
            "response_text": request.response_text,
            "critic_reports": [],
            "disagreements": []
        })
        
        verdict = state.get("final_verdict") or {}
        reports = state.get("critic_reports", [])
        disagreements = state.get("disagreements", [])
        
        record_id = save_arbitration(
            prompt=request.prompt,
            response_text=request.response_text,
            verdict=verdict,
            critic_reports=reports,
            disagreements=disagreements
        )
        
        return ArbitrationResponse(
            arbitration_id=record_id,
            overall_quality_score=verdict.get("overall_quality_score", 0),
            confidence_score=verdict.get("confidence_score", 0.0),
            executive_summary=verdict.get("executive_summary", ""),
            confirmed_issues_count=len(verdict.get("confirmed_issues", [])),
            dismissed_flags_count=len(verdict.get("dismissed_flags", [])),
            verdict=verdict,
            critic_reports=[r.model_dump() if hasattr(r, "model_dump") else r for r in reports],
            disagreements=[d.model_dump() if hasattr(d, "model_dump") else d for d in disagreements]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Arbitration failed: {str(e)}"
        )

@app.post("/v1/arbitrate/batch", status_code=status.HTTP_201_CREATED)
def arbitrate_batch(request: BatchArbitrationRequest):
    """Processes multiple outputs sequentially and logs all verdicts."""
    batch_results = []
    for item in request.items:
        res = arbitrate_single(item)
        batch_results.append(res)
    return {"total_processed": len(batch_results), "results": batch_results}

@app.get("/v1/arbitrations/{arbitration_id}")
def get_arbitration(arbitration_id: str):
    """Fetches an existing arbitration audit record by ID."""
    record = get_arbitration_by_id(arbitration_id)
    if not record:
        raise HTTPException(status_code=404, detail="Arbitration record not found")
    return record

@app.get("/v1/analytics")
def get_analytics():
    """Returns meta-analytics on critic performance, overrules, and agreement rates."""
    return calculate_critic_analytics()