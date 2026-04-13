import os
import gradio as gr
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_client():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def classify_email(email_text: str):
    if not email_text.strip():
        return "⚠️ Please enter an email.", "", ""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an email classifier. Reply in exactly this format:\nCategory: spam/important/normal\nPriority: high/medium/low\nResponse: your short reply"},
                {"role": "user", "content": f"Classify this email: {email_text}"},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        category, priority, suggestion = "", "", ""
        for line in raw.splitlines():
            if line.lower().startswith("category:"):
                category = line.split(":", 1)[1].strip().capitalize()
            elif line.lower().startswith("priority:"):
                priority = line.split(":", 1)[1].strip().capitalize()
            elif line.lower().startswith("response:"):
                suggestion = line.split(":", 1)[1].strip()
        return f"## 📂 {category}", f"## 🚨 {priority}", f"### 💬 {suggestion}"
    except Exception as e:
        return f"Error: {str(e)}", "", ""

EXAMPLES = [
    ["Win a FREE iPhone! Click now to claim your prize!!!"],
    ["Hi, please find attached the Q3 financial report for your review."],
    ["Your Amazon order has been shipped and will arrive tomorrow."],
    ["Can we reschedule our 3pm meeting to 4pm today?"],
]

with gr.Blocks(title="📧 Email Triage AI") as demo:
    gr.Markdown("# 📧 Email Triage AI\nPowered by Llama 3.1 · Paste your email on the left, get results on the right.")

    with gr.Row():
        # LEFT SIDE - Input
        with gr.Column(scale=1):
            gr.Markdown("### ✉️ Your Email")
            email_input = gr.Textbox(
                label="Paste email here",
                placeholder="Paste your email content here...",
                lines=15,
                max_lines=20,
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Classify", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            gr.Examples(
                examples=EXAMPLES,
                inputs=email_input,
                label="📋 Try an Example",
            )

        # RIGHT SIDE - Output
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            category_out = gr.Markdown(value="", label="📂 Category")
            priority_out = gr.Markdown(value="", label="🚨 Priority")
            response_out = gr.Markdown(value="", label="💬 Suggested Response")
            gr.Markdown("---")
            gr.Markdown("💡 **How it works:** The AI reads your email and classifies it into spam/important/normal, assigns a priority level, and suggests a short reply.")

    submit_btn.click(
        fn=classify_email,
        inputs=email_input,
        outputs=[category_out, priority_out, response_out],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[email_input, category_out, priority_out, response_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)