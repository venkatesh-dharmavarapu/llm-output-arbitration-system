import json
import time
from src.graph.workflow import arbitration_graph

prompt = "Explain why Apollo 11 was significant, who walked on the moon, and when it happened."
response = (
    "Apollo 11 was a historic spaceflight that landed the first humans on the Moon. "
    "American astronauts Neil Armstrong and Buzz Aldrin stepped onto the lunar surface on July 20, 1969. "
    "This mission concluded the space race decisively, as Neil Armstrong famously uttered: "
    "'That's one small step for man, one giant leap for the Soviet Union.'"
)

print("Starting full pipeline arbitration (Fan-out -> Critics -> Collector -> Adjudicator)...")
start = time.time()

res = arbitration_graph.invoke({
    "prompt": prompt,
    "response_text": response,
    "critic_reports": [],
    "disagreements": []
})

print(f"\nCompleted in {time.time() - start:.2f}s")
print("\n=== FINAL ADJUDICATED VERDICT ===")
print(json.dumps(res["final_verdict"], indent=2))