import sys
from pathlib import Path

# Add project root to sys.path so Python recognizes 'src' and 'ui'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.graph.workflow import arbitration_graph
from ui.highlighter import render_annotated_text
from ui.batch_runner import run_batch_arbitration

st.set_page_config(
    page_title="LLM Output Arbitration System",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LLM Output Arbitration System")
st.caption("Multi-agent evaluation pipeline: Parallel critics -> Disagreement detection -> Chief Adjudication")

tab_single, tab_batch = st.tabs(["Single Output Arbitration", "Batch Mode"])

# --- Tab 1: Single Arbitration ---
with tab_single:
    col_input, col_preset = st.columns([3, 1])
    
    with col_preset:
        st.subheader("Presets")
        preset_choice = st.selectbox(
            "Load an example test case:",
            ["Custom", "Historical Misquote (Plant)", "Primary Colors Contradiction", "Clean / Flawless Response"]
        )
        
    default_prompt = ""
    default_response = ""
    
    if preset_choice == "Historical Misquote (Plant)":
        default_prompt = "Explain why Apollo 11 was significant, who walked on the moon, and when it happened."
        default_response = (
            "Apollo 11 was a historic spaceflight that landed the first humans on the Moon. "
            "American astronauts Neil Armstrong and Buzz Aldrin stepped onto the lunar surface on July 20, 1969. "
            "This mission concluded the space race decisively, as Neil Armstrong famously uttered: "
            "'That's one small step for man, one giant leap for the Soviet Union.'"
        )
    elif preset_choice == "Primary Colors Contradiction":
        default_prompt = "Name 3 primary colors and state why they cannot be created by mixing other colors."
        default_response = "The 3 primary colors are Red, Blue, and Green. You can make Red by mixing Orange and Pink."
    elif preset_choice == "Clean / Flawless Response":
        default_prompt = "What is the capital of France and what river runs through it?"
        default_response = "The capital of France is Paris. The Seine River flows through the heart of the city."

    with col_input:
        prompt_input = st.text_area("Original Prompt:", value=default_prompt, height=100)
        response_input = st.text_area("Candidate LLM Output to Audit:", value=default_response, height=140)

    if st.button("Arbitrate Output", type="primary", use_container_width=True):
        if not prompt_input or not response_input:
            st.error("Please provide both a prompt and candidate output.")
        else:
            with st.spinner("Dispatching parallel critics and adjudicating..."):
                state = arbitration_graph.invoke({
                    "prompt": prompt_input,
                    "response_text": response_input,
                    "critic_reports": [],
                    "disagreements": []
                })
                
            verdict = state.get("final_verdict") or {}
            reports = state.get("critic_reports", [])
            disagreements = state.get("disagreements", [])

            st.divider()
            
            # Top Score Header
            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Adjudicated Score", f"{verdict.get('overall_quality_score', 0)} / 10")
            score_col2.metric("Adjudication Confidence", f"{verdict.get('confidence_score', 0.0) * 100:.0f}%")
            score_col3.metric("Inter-Critic Conflicts", len(disagreements))

            # Annotated Text Display
            st.subheader("Inline Issue Annotations")
            annotated_html = render_annotated_text(response_input, verdict.get("confirmed_issues", []))
            st.markdown(annotated_html, unsafe_allow_html=True)
            st.caption("Hover over highlighted text to inspect evidence and reasoning.")

            # Executive Summary
            st.info(f"**Executive Summary:** {verdict.get('executive_summary', 'N/A')}")

            # Critic Matrix
            st.subheader("Critic Assessment Matrix")
            critic_cols = st.columns(len(reports))
            for idx, r in enumerate(reports):
                with critic_cols[idx]:
                    st.markdown(f"#### {r.dimension.replace('_', ' ').title()}")
                    st.write(f"**Model:** `{r.model_name}`")
                    st.metric("Score", f"{r.score}/5", delta=f"{r.confidence*100:.0f}% conf")
                    st.caption(r.reasoning_summary)
                    if r.issues:
                        with st.expander(f"Issues ({len(r.issues)})"):
                            for issue in r.issues:
                                st.markdown(f"- **[{issue.severity.upper()}]** {issue.problem}")

            # Issues Detail Breakdown
            st.subheader("Adjudication Breakdown")
            c_col, d_col = st.columns(2)
            with c_col:
                st.markdown("##### Confirmed Issues")
                confirmed = verdict.get("confirmed_issues", [])
                if not confirmed:
                    st.success("No confirmed flaws.")
                for c in confirmed:
                    st.warning(f"**Quote:** \"{c.get('quote')}\"\n\n**Impact:** {c.get('dimension').upper()} | **Severity:** {c.get('severity')}\n\n**Reasoning:** {c.get('evidence_reasoning')}")

            with d_col:
                st.markdown("##### Dismissed Flags (Overruled Critics)")
                dismissed = verdict.get("dismissed_flags", [])
                if not dismissed:
                    st.write("No critic flags were overruled.")
                for d in dismissed:
                    st.info(f"**Quote:** \"{d.get('quote')}\"\n\n**Critic:** {d.get('raised_by_dimension')}\n\n**Dismissal Reason:** {d.get('reason_for_dismissal')}")

# --- Tab 2: Batch Mode ---
with tab_batch:
    st.subheader("Batch Arbitration Evaluation")
    st.write("Submit multiple prompts and responses to evaluate system performance at scale.")
    
    if st.button("Run Preloaded Batch Evaluation"):
        test_batch = [
            {
                "prompt": "Name 3 primary colors and why they cannot be mixed.",
                "response_text": "The 3 primary colors are Red, Blue, and Green. You can make Red by mixing Orange and Pink."
            },
            {
                "prompt": "Explain why Apollo 11 was significant.",
                "response_text": "Apollo 11 was a historic flight landing Neil Armstrong and Buzz Aldrin on July 20, 1969, concluding the space race for the Soviet Union."
            },
            {
                "prompt": "What is the boiling point of water at sea level in Celsius?",
                "response_text": "Water boils at 100°C (212°F) at standard atmospheric pressure at sea level."
            }
        ]
        
        with st.spinner("Processing batch arbitration across pipeline..."):
            batch_results = run_batch_arbitration(test_batch)
            st.table(batch_results)