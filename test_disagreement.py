import time
from src.graph.workflow import arbitration_graph

# This response is logically structured and complete, but contains a subtle factual hallucination
disputed_prompt = "Explain why Apollo 11 was significant, who walked on the moon, and when it happened."
disputed_output = (
    "Apollo 11 was a historic spaceflight that successfully landed the first humans on the Moon. "
    "American astronauts Neil Armstrong and Buzz Aldrin stepped onto the lunar surface on July 20, 1969, "
    "while Michael Collins orbited in the command module. This mission answered all objectives and concluded "
    "the space race decisively, as Neil Armstrong famously uttered: 'That's one small step for man, one giant leap for the Soviet Union.'"
)

print("Running arbitration on conflicting test case...")
start = time.time()

state = arbitration_graph.invoke({
    "prompt": disputed_prompt,
    "response_text": disputed_output,
    "critic_reports": [],
    "disagreements": []
})

print(f"Completed in {time.time() - start:.2f}s")
print("\n--- Individual Critic Scores ---")
for report in state["critic_reports"]:
    print(f"[{report.dimension.value}] Score: {report.score}/5 | Issues: {len(report.issues)} | Model: {report.model_name}")

print("\n--- Disagreements Detected ---")
if not state["disagreements"]:
    print("No disagreements triggered.")
else:
    for d in state["disagreements"]:
        print(f"- Type: {d.disagreement_type}")
        print(f"  Description: {d.description}")
        print(f"  Involved: {', '.join(d.critics_involved)}")