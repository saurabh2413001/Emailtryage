import sys
import os
import requests
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.environment import EmailEnv
from env.models import Action
from env.grader import grade

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------- PROMPT ---------------- #

def build_prompt(email):
    return f"""You are an AI email assistant. Respond ONLY in this exact format, nothing else:

Category: spam
Priority: high
Response: short reply

Now classify this email:
Email: {email}"""

# ---------------- API CALL ---------------- #

def query_llm(prompt):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an email classifier. Always reply in exactly 3 lines: Category, Priority, Response."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
    )

    print(f"HTTP Status: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: {response.text}")
        exit()

    return response.json()

# ---------------- PARSER ---------------- #

def parse_output(output_text):
    text = output_text.lower()

    category = "normal"
    priority = "medium"
    response = "Okay."

    if "spam" in text:
        category = "spam"
    elif "important" in text:
        category = "important"

    if "high" in text:
        priority = "high"
    elif "low" in text:
        priority = "low"

    if "response:" in text:
        response = output_text.split("Response:")[-1].strip()
    else:
        response = output_text.strip()

    return Action(category=category, priority=priority, response=response)

# ---------------- MAIN ---------------- #

env = EmailEnv()
obs = env.reset()

print("\n📧 Email to classify:", obs.email_text)

prompt = build_prompt(obs.email_text)
result = query_llm(prompt)

try:
    output_text = result["choices"][0]["message"]["content"]
    print(f"\n✅ Generated text:\n{output_text}")
except (KeyError, IndexError, TypeError) as e:
    print("❌ Unexpected response format:", result)
    exit()

action = parse_output(output_text)
obs, reward, done, _ = env.step(action)
truth = env.state()
score = grade(action, truth)

print("\n🤖 AI Output:", action)
print("🎯 Reward:", reward)
print("📊 Score:", score)