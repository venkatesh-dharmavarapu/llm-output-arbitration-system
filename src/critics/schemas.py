from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class EvaluationDimension(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMPLETENESS = "completeness"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FlaggedIssue(BaseModel):
    quote: str = Field(description="Exact snippet from text containing the issue.")
    problem: str = Field(description="Explanation of why this is an issue.")
    severity: SeverityLevel = Field(description="Severity rating.")

class CriticReport(BaseModel):
    dimension: EvaluationDimension
    model_name: str
    score: int = Field(ge=1, le=5, description="1 is unusable, 5 is flawless.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence level.")
    issues: List[FlaggedIssue] = Field(default_factory=list)
    reasoning_summary: str = Field(description="High-level evaluation explanation.")