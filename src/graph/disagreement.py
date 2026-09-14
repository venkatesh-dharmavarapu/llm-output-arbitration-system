from typing import List
from src.critics.schemas import CriticReport, SeverityLevel
from src.graph.state import DisagreementRecord

def detect_disagreements(reports: List[CriticReport]) -> List[DisagreementRecord]:
    disagreements: List[DisagreementRecord] = []
    if len(reports) < 2:
        return disagreements

    # 1. Score divergence (>= 2 points difference)
    scores = [(r.dimension.value, r.score, r.model_name) for r in reports]
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            dim1, score1, model1 = scores[i]
            dim2, score2, model2 = scores[j]
            if abs(score1 - score2) >= 2:
                disagreements.append(
                    DisagreementRecord(
                        disagreement_type="score_variance",
                        description=f"Score divergence: {dim1} scored {score1} while {dim2} scored {score2} (diff: {abs(score1 - score2)}).",
                        critics_involved=[model1, model2]
                    )
                )

    # 2. Critical Issue vs High Overall Score
    for report in reports:
        has_critical = any(issue.severity == SeverityLevel.CRITICAL for issue in report.issues)
        if has_critical:
            for other in reports:
                if other.dimension != report.dimension and other.score >= 4:
                    disagreements.append(
                        DisagreementRecord(
                            disagreement_type="severity_conflict",
                            description=f"{report.dimension} flagged critical issues, but {other.dimension} awarded a high score of {other.score}.",
                            critics_involved=[report.model_name, other.model_name]
                        )
                    )

    # 3. Issue Count Disparity
    issue_counts = {r.dimension.value: len(r.issues) for r in reports}
    for dim, count in issue_counts.items():
        if count >= 2 and all(issue_counts[other] == 0 for other in issue_counts if other != dim):
            disagreements.append(
                DisagreementRecord(
                    disagreement_type="isolated_issue",
                    description=f"Only {dim} identified issues ({count} found); other critics detected none.",
                    critics_involved=[dim]
                )
            )

    return disagreements