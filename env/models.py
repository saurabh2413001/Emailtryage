from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str
    sender: str
    urgency_hint: str  # hint for priority (low/high)

class Action(BaseModel):
    category: str      # spam / important / normal
    priority: str      # low / medium / high
    response: str      # agent's reply

class Reward(BaseModel):
    score: float