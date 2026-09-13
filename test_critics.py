from src.critics.runners import get_accuracy_critic, get_logic_critic, get_completeness_critic

prompt = "Name 3 primary colors and state why they cannot be created by mixing other colors."
flawed_response = "The 3 primary colors are Red, Blue, and Green. You can make Red by mixing Orange and Pink."

print("--> 1. Running Groq Accuracy Critic...")
print(get_accuracy_critic(prompt, flawed_response).model_dump_json(indent=2))

print("\n--> 2. Running Groq Logic Critic...")
print(get_logic_critic(prompt, flawed_response).model_dump_json(indent=2))

print("\n--> 3. Running Local Ollama Completeness Critic...")
print(get_completeness_critic(prompt, flawed_response).model_dump_json(indent=2))