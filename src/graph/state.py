import operator
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from src.critics.schemas import CriticReport

class DisagreementRecord(BaseModel):
    disagreement_type: str = Field(description="Type: 'score_variance', 'severity_conflict', or 'isolated_issue'.")
    description: str = Field(description="Explanation of the conflict.")
    critics_involved: List[str] = Field(description="List of model names involved.")

class ArbitrationState(BaseModel):
    prompt: str
    response_text: str
    # Parallel fan-in: appends reports as each critic node completes
    critic_reports: Annotated[List[CriticReport], operator.add] = Field(default_factory=list)
    disagreements: List[DisagreementRecord] = Field(default_factory=list)
    is_unanimous_pass: bool = False
    final_verdict: Optional[dict] = None