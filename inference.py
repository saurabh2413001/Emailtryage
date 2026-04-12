import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an email classifier. Reply in exactly this format:\nCategory: spam/important/normal\nPriority: high/medium/low\nResponse: your short reply"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=100,
        temperature=0.1
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(run_inference("Hello from OpenEnv!"))