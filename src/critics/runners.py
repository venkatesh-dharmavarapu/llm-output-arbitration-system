import os
from dotenv import load_dotenv
import instructor
from groq import Groq
from src.critics.schemas import CriticReport, EvaluationDimension

load_dotenv()

groq_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON
)

BASE_DIRECTIVE = (
    "Be direct and concise. Limit problem explanations to at most 2 sentences. "
    "Limit reasoning_summary to 2 sentences. Ensure the JSON document is complete."
)

def get_accuracy_critic(prompt: str, response_text: str) -> CriticReport:
    """Evaluates factual accuracy using GPT-OSS 120B."""
    system_prompt = (
        f"You are an expert Fact-Checking Critic. Check for factual correctness, "
        f"accurate dates, valid citations, and hallucinations. {BASE_DIRECTIVE}"
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=CriticReport,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    report.dimension = EvaluationDimension.FACTUAL_ACCURACY
    report.model_name = "gpt-oss-120b (Groq)"
    return report

def get_logic_critic(prompt: str, response_text: str) -> CriticReport:
    """Evaluates logical reasoning using Qwen 3.8 27B."""
    system_prompt = (
        f"You are a Formal Logic Critic. Check whether conclusions follow logically, "
        f"and flag non-sequiturs, contradictions, or invalid leaps. {BASE_DIRECTIVE}"
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        response_model=CriticReport,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    report.dimension = EvaluationDimension.LOGICAL_CONSISTENCY
    report.model_name = "qwen3.8-27b (Groq)"
    return report

def get_completeness_critic(prompt: str, response_text: str) -> CriticReport:
    """Evaluates question coverage using Qwen 3.6 27B."""
    system_prompt = (
        f"You are a Completeness Critic. Check if the response answers every part "
        f"of the prompt without omitting instructions. {BASE_DIRECTIVE}"
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        response_model=CriticReport,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    report.dimension = EvaluationDimension.COMPLETENESS
    report.model_name = "qwen3.6-27b (Groq)"
    return report