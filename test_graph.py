import time
from src.graph.workflow import arbitration_graph

sample_prompt = "Name 3 primary colors and state why they cannot be created by mixing other colors."
flawed_output = "The 3 primary colors are Red, Blue, and Green. You can make Red by mixing Orange and Pink."

print("Starting LangGraph parallel fan-out...")
start_time = time.time()

initial_state = {
    "prompt": sample_prompt,
    "response_text": flawed_output,
    "critic_reports": [],
    "disagreements": []
}

final_state = arbitration_graph.invoke(initial_state)
elapsed = time.time() - start_time

print(f"\nExecution completed in {elapsed:.2f} seconds.")
print(f"Total Critic Reports Collected: {len(final_state['critic_reports'])}")
print(f"Unanimous Pass?: {final_state['is_unanimous_pass']}")
print(f"Disagreements Detected: {len(final_state['disagreements'])}")

for d in final_state['disagreements']:
    print(f" - [{d.disagreement_type}] {d.description}")