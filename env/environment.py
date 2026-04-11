import random
from env.models import Observation, Action

class EmailEnv:
    def __init__(self):
        self.emails = []
        self.current = None
        self.done = False

    def reset(self):
        # Sample emails for simulation
        self.emails = [
            {"text": "Win a free iPhone now!!!", "sender": "unknown", "type": "spam"},
            {"text": "Project deadline is tomorrow", "sender": "boss", "type": "important"},
            {"text": "Lunch at 2?", "sender": "friend", "type": "normal"},
        ]
        self.current = random.choice(self.emails)
        self.done = False

        return Observation(
            email_text=self.current["text"],
            sender=self.current["sender"],
            urgency_hint="high" if self.current["type"] == "important" else "low"
        )

    def step(self, action: Action):
        reward = 0.0

        # Classification reward
        if action.category == self.current["type"]:
            reward += 0.4

        # Priority reward
        if self.current["type"] == "important" and action.priority == "high":
            reward += 0.3

        # Response reward (basic length check)
        if len(action.response) > 5:
            reward += 0.3

        self.done = True

        return (
            Observation(email_text="", sender="", urgency_hint=""),
            reward,
            self.done,
            {}
        )

    def state(self):
        return self.current