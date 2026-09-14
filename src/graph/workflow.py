from langgraph.graph import StateGraph, START, END
from src.graph.state import ArbitrationState
from src.critics.schemas import CriticReport, EvaluationDimension
from src.critics.runners import (
    get_accuracy_critic,
    get_logic_critic,
    get_completeness_critic
)
from src.graph.disagreement import detect_disagreements

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
            model_name="gpt-oss-20b (Degraded)",
            score=3,
            confidence=0.0,
            issues=[],
            reasoning_summary=f"Critic unavailable: {str(e)[:120]}"
        )
        return {"critic_reports": [fallback]}

def collector_and_disagreement_node(state: ArbitrationState):
    reports = state.critic_reports
    # Only calculate disagreements across successful critiques
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

builder = StateGraph(ArbitrationState)

builder.add_node("accuracy_critic", accuracy_node)
builder.add_node("logic_critic", logic_node)
builder.add_node("completeness_critic", completeness_node)
builder.add_node("collector", collector_and_disagreement_node)

builder.add_edge(START, "accuracy_critic")
builder.add_edge(START, "logic_critic")
builder.add_edge(START, "completeness_critic")

builder.add_edge("accuracy_critic", "collector")
builder.add_edge("logic_critic", "collector")
builder.add_edge("completeness_critic", "collector")

builder.add_edge("collector", END)

arbitration_graph = builder.compile()