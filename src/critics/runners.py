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

def get_accuracy_critic(prompt: str, response_text: str) -> CriticReport:
    """Evaluates factual accuracy using GPT-OSS 120B."""
    system_prompt = (
        "You are an expert Fact-Checking Critic. Check for factual correctness, "
        "accurate dates, valid citations, and hallucinations. Return a structured critique."
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=CriticReport,
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
        "You are a Formal Logic Critic. Check whether conclusions follow logically, "
        "and flag non-sequiturs, internal contradictions, or invalid leaps. Return a structured critique."
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        response_model=CriticReport,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    report.dimension = EvaluationDimension.LOGICAL_CONSISTENCY
    report.model_name = "qwen3.8-27b (Groq)"
    return report

def get_completeness_critic(prompt: str, response_text: str) -> CriticReport:
    """Evaluates question coverage using GPT-OSS 20B."""
    system_prompt = (
        "You are a Completeness Critic. Check if the response answers every part "
        "of the original prompt without missing edge cases or instructions. Return a structured critique."
    )
    content = f"User Prompt:\n{prompt}\n\nLLM Response to Audit:\n{response_text}"
    
    report = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        response_model=CriticReport,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    report.dimension = EvaluationDimension.COMPLETENESS
    report.model_name = "gpt-oss-20b (Groq)"
    return report