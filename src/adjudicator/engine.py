import os
import json
from typing import List
from groq import Groq
import instructor
from src.critics.schemas import CriticReport
from src.graph.state import DisagreementRecord
from src.adjudicator.schemas import ArbitrationVerdict

client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON
)

def adjudicate(
    prompt: str,
    response_text: str,
    critic_reports: List[CriticReport],
    disagreements: List[DisagreementRecord]
) -> ArbitrationVerdict:
    """Weighs critic evidence, resolves conflicts, and returns a final scored verdict."""

    # Filter only successful critiques for adjudication
    reports_data = [r.model_dump() for r in critic_reports if r.confidence > 0.0]
    disagreements_data = [d.model_dump() for d in disagreements]

    system_prompt = (
        "You are the Chief LLM Output Adjudicator. Your job is to resolve disagreements between "
        "three specialized critic models (Accuracy, Logic, Completeness) evaluating an LLM output.\n"
        "Instructions:\n"
        "1. Examine the original prompt and the candidate LLM response.\n"
        "2. Review each critic's findings and any detected disagreements.\n"
        "3. Confirm genuine issues and assign fair severity ratings.\n"
        "4. Dismiss nitpicks or false-positive critiques with clear reasoning.\n"
        "5. Keep the executive summary direct and under 3 sentences."
    )

    user_content = (
        f"### USER PROMPT:\n{prompt}\n\n"
        f"### CANDIDATE LLM RESPONSE:\n{response_text}\n\n"
        f"### CRITIC REPORTS:\n{json.dumps(reports_data, indent=2)}\n\n"
        f"### DETECTED DISAGREEMENTS:\n{json.dumps(disagreements_data, indent=2)}\n\n"
        "Return the structured ArbitrationVerdict."
    )

    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=ArbitrationVerdict,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )