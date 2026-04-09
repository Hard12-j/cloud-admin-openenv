import random

class State:
    def __init__(self, steps, difficulty):
        self.step_count = steps
        self.difficulty = difficulty
        self.resources = {"i-temp": {"status": "running", "tags": {"purpose": "temporary"}}}
        self.users = {"hacker123": {"status": "active"}}

def calculate_reward(state):
    score = 0.2
    score -= (state.step_count * 0.01)
    
    if state.difficulty == "easy":
        status = state.resources["i-temp"]["status"]
        if status == "stopped":
            score += 0.7
        elif status == "terminated":
            score += 0.5
    
    # ... other diffs ...
    
    score += random.uniform(-0.01, 0.01)
    return max(0.01, min(0.99, round(score, 3)))

# Case: Fail after 15 steps
s = State(15, "hard")
results = [calculate_reward(s) for _ in range(10)]
print(f"Hard Fail (15 steps): {results}")

# Case: Success after 5 steps
s = State(5, "easy")
s.resources["i-temp"]["status"] = "stopped"
results = [calculate_reward(s) for _ in range(10)]
print(f"Easy Success (5 steps): {results}")
