import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

app = FastAPI()

EMAILS = [
    {"email_text": "Win a free iPhone now!!!", "sender": "unknown", "type": "spam"},
    {"email_text": "Project deadline is tomorrow", "sender": "boss", "type": "important"},
    {"email_text": "Lunch at 2?", "sender": "friend", "type": "normal"},
]

current_email = None

class ResetRequest(BaseModel):
    seed: int = 0

class StepRequest(BaseModel):
    action: str

def run_inference(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an email classifier. Reply in exactly this format:\nCategory: spam/important/normal\nPriority: high/medium/low\nResponse: your short reply"
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.1
    )
    return response.choices[0].message.content

@app.post("/reset")
def reset(request: ResetRequest):
    global current_email
    random.seed(request.seed)
    current_email = random.choice(EMAILS)
    return {
        "email_text": current_email["email_text"],
        "sender": current_email["sender"],
        "urgency_hint": "high" if current_email["type"] == "important" else "low"
    }

@app.post("/step")
def step(request: StepRequest):
    result = run_inference(request.action)
    return {
        "output": result,
        "reward": 1.0,
        "done": True
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)