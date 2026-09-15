from typing import List
from pydantic import BaseModel, Field
from src.critics.schemas import SeverityLevel

class AdjudicatedIssue(BaseModel):
    quote: str = Field(description="Exact snippet from the text containing the verified issue.")
    dimension: str = Field(description="The primary dimension impacted (accuracy, logic, or completeness).")
    severity: SeverityLevel = Field(description="Adjudicated severity level.")
    evidence_reasoning: str = Field(description="Why this issue was confirmed based on the evidence.")

class DismissedIssue(BaseModel):
    quote: str = Field(description="Snippet flagged by a critic.")
    raised_by_dimension: str = Field(description="Which critic dimension raised the flag.")
    reason_for_dismissal: str = Field(description="Why the adjudicator overruled the critic's flag.")

class ArbitrationVerdict(BaseModel):
    overall_quality_score: int = Field(ge=1, le=10, description="Overall score from 1 (unusable) to 10 (flawless).")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Adjudicator's confidence in this final assessment.")
    executive_summary: str = Field(description="Concise paragraph synthesizing the final verdict.")
    confirmed_issues: List[AdjudicatedIssue] = Field(default_factory=list)
    dismissed_flags: List[DismissedIssue] = Field(default_factory=list)