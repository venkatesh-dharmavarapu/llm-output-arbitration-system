# ⚖️ LLM Output Arbitration System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)

An enterprise-grade, multi-agent evaluation framework designed to audit and arbitrate candidate LLM outputs for high-stakes enterprise applications. Built with a heterogeneous multi-critic architecture, automated disagreement detection, and evidence-based synthesis.

---

## 🏗️ Architecture Overview

The system uses a parallel **fan-out / fan-in** pipeline powered by **LangGraph** to concurrently evaluate text across three dimensions before a Chief Adjudicator resolves discrepancies.

```mermaid
graph TD
    User([User Prompt & Candidate Output]) --> Dispatch{LangGraph Parallel Dispatch}
    
    Dispatch -->|Async Fan-Out| C1[Accuracy Critic<br/><i>GPT-OSS 120B</i>]
    Dispatch -->|Async Fan-Out| C2[Logic Critic<br/><i>Qwen 3.8 27B</i>]
    Dispatch -->|Async Fan-Out| C3[Completeness Critic<br/><i>GPT-OSS 20B</i>]
    
    C1 --> Collector[Fan-In Collector & Disagreement Engine]
    C2 --> Collector
    C3 --> Collector
    
    Collector -->|Clean Pass Short-Circuit| Output([Final Verdict])
    Collector -->|Conflicts / Flaws Detected| Adjudicator[Chief Adjudicator Agent<br/><i>GPT-OSS 120B</i>]
    
    Adjudicator --> Output
    Output --> Storage[(SQLite Audit Trail)]
    Output --> UI[Streamlit Verdict Explorer]
    Output --> API[FastAPI Endpoints]