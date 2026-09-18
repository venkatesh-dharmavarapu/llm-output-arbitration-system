import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage.db import init_db, save_arbitration
from src.graph.workflow import arbitration_graph

init_db()

SEED_CASES = [
    {
        "prompt": "Name 3 primary colors and state why they cannot be created by mixing other colors.",
        "response_text": "The 3 primary colors are Red, Blue, and Green. You can make Red by mixing Orange and Pink."
    },
    {
        "prompt": "Explain why Apollo 11 was significant, who walked on the moon, and when it happened.",
        "response_text": (
            "Apollo 11 landed humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin walked "
            "on the surface. Armstrong famously declared: 'One small step for man, one giant leap for the Soviet Union.'"
        )
    },
    {
        "prompt": "If all mammals are warm-blooded, and all whales are mammals, what can we deduce about whales?",
        "response_text": (
            "Whales are aquatic mammals. Because ocean water is freezing, whales must be cold-blooded to regulate their temperature."
        )
    },
    {
        "prompt": "What is the capital of France and what river flows through it?",
        "response_text": "The capital of France is Paris, and the Seine River flows directly through it."
    }
]

def run_seed():
    print("🌱 Seeding arbitration audit database...")
    for idx, case in enumerate(SEED_CASES, start=1):
        print(f"[{idx}/{len(SEED_CASES)}] Auditing: {case['prompt'][:45]}...")
        state = arbitration_graph.invoke({
            "prompt": case["prompt"],
            "response_text": case["response_text"],
            "critic_reports": [],
            "disagreements": []
        })
        record_id = save_arbitration(
            prompt=case["prompt"],
            response_text=case["response_text"],
            verdict=state.get("final_verdict") or {},
            critic_reports=state.get("critic_reports", []),
            disagreements=state.get("disagreements", [])
        )
        print(f"   -> Stored record: {record_id}")
    print("✅ Seeding complete! Database is populated with realistic audit history.")

if __name__ == "__main__":
    run_seed()