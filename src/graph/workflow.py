from langgraph.graph import StateGraph, START, END
from src.graph.state import ArbitrationState
from src.critics.schemas import CriticReport, EvaluationDimension
from src.critics.runners import (
    get_accuracy_critic,
    get_logic_critic,
    get_completeness_critic
)
from src.graph.disagreement import detect_disagreements
from src.adjudicator.engine import adjudicate

# --- Critic Nodes with Fault Tolerance ---

def accuracy_node(state: ArbitrationState):
    try:
        report = get_accuracy_critic(state.prompt, state.response_text)
        return {"critic_reports": [report]}
    except Exception as e:
        fallback = CriticReport(
            dimension=EvaluationDimension.FACTUAL_ACCURACY,
            model_name="gpt-oss-120b (Degraded)",
            score=3,
            confidence=0.0,
            issues=[],
            reasoning_summary=f"Critic unavailable: {str(e)[:120]}"
        )
        return {"critic_reports": [fallback]}

def logic_node(state: ArbitrationState):
    try:
        report = get_logic_critic(state.prompt, state.response_text)
        return {"critic_reports": [report]}
    except Exception as e:
        fallback = CriticReport(
            dimension=EvaluationDimension.LOGICAL_CONSISTENCY,
            model_name="qwen3.8-27b (Degraded)",
            score=3,
            confidence=0.0,
            issues=[],
            reasoning_summary=f"Critic unavailable: {str(e)[:120]}"
        )
        return {"critic_reports": [fallback]}

def completeness_node(state: ArbitrationState):
    try:
        report = get_completeness_critic(state.prompt, state.response_text)
        return {"critic_reports": [report]}
    except Exception as e:
        fallback = CriticReport(
            dimension=EvaluationDimension.COMPLETENESS,
            model_name="qwen3.6-27b (Degraded)",
            score=3,
            confidence=0.0,
            issues=[],
            reasoning_summary=f"Critic unavailable: {str(e)[:120]}"
        )
        return {"critic_reports": [fallback]}

# --- Fan-In Collector & Disagreement Detector ---

def collector_and_disagreement_node(state: ArbitrationState):
    reports = state.critic_reports
    valid_reports = [r for r in reports if r.confidence > 0.0]
    disagreements = detect_disagreements(valid_reports)
    all_clean = (
        len(valid_reports) >= 2
        and all(r.score == 5 and len(r.issues) == 0 for r in valid_reports)
    )
    return {
        "disagreements": disagreements,
        "is_unanimous_pass": all_clean
    }

# --- Chief Adjudicator Node ---

def adjudication_node(state: ArbitrationState):
    if state.is_unanimous_pass:
        verdict = {
            "overall_quality_score": 10,
            "confidence_score": 0.99,
            "executive_summary": "All critics unanimously verified this output as accurate, logical, and complete with zero issues.",
            "confirmed_issues": [],
            "dismissed_flags": []
        }
    else:
        verdict_obj = adjudicate(
            prompt=state.prompt,
            response_text=state.response_text,
            critic_reports=state.critic_reports,
            disagreements=state.disagreements
        )
        verdict = verdict_obj.model_dump()
    return {"final_verdict": verdict}

# --- Graph Assembly ---

builder = StateGraph(ArbitrationState)

builder.add_node("accuracy_critic", accuracy_node)
builder.add_node("logic_critic", logic_node)
builder.add_node("completeness_critic", completeness_node)
builder.add_node("collector", collector_and_disagreement_node)
builder.add_node("adjudicator", adjudication_node)

# Parallel Fan-Out
builder.add_edge(START, "accuracy_critic")
builder.add_edge(START, "logic_critic")
builder.add_edge(START, "completeness_critic")

# Parallel Fan-In
builder.add_edge("accuracy_critic", "collector")
builder.add_edge("logic_critic", "collector")
builder.add_edge("completeness_critic", "collector")

# Adjudication & Termination
builder.add_edge("collector", "adjudicator")
builder.add_edge("adjudicator", END)

arbitration_graph = builder.compile()