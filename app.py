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
        return (
            "<div class='result-card empty'><span class='result-label'>CATEGORY</span><span class='result-value'>—</span></div>",
            "<div class='result-card empty'><span class='result-label'>PRIORITY</span><span class='result-value'>—</span></div>",
            "<div class='result-card empty'><span class='result-label'>SUGGESTED REPLY</span><span class='result-value'>Please enter an email.</span></div>",
        )
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

        cat_colors = {"Spam": "#ef4444", "Important": "#10b981", "Normal": "#3b82f6"}
        pri_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        cat_icons = {"Spam": "🚫", "Important": "⭐", "Normal": "📬"}
        pri_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

        cat_color = cat_colors.get(category, "#94a3b8")
        pri_color = pri_colors.get(priority, "#94a3b8")
        cat_icon = cat_icons.get(category, "📂")
        pri_icon = pri_icons.get(priority, "🚨")

        cat_html = f"<div class='result-card' style='border-left: 4px solid {cat_color};'><span class='result-label'>CATEGORY</span><span class='result-value' style='color:{cat_color};'>{cat_icon} {category}</span></div>"
        pri_html = f"<div class='result-card' style='border-left: 4px solid {pri_color};'><span class='result-label'>PRIORITY</span><span class='result-value' style='color:{pri_color};'>{pri_icon} {priority}</span></div>"
        res_html = f"<div class='result-card' style='border-left: 4px solid #818cf8;'><span class='result-label'>SUGGESTED REPLY</span><span class='result-value reply-text'>💬 {suggestion}</span></div>"

        return cat_html, pri_html, res_html

    except Exception as e:
        err = f"<div class='result-card error'><span class='result-label'>ERROR</span><span class='result-value' style='color:#ef4444; font-size:0.9rem;'>❌ {str(e)}</span></div>"
        return err, err, err


EXAMPLES = [
    ["Win a FREE iPhone! Click now to claim your prize!!!"],
    ["Hi, please find attached the Q3 financial report for your review."],
    ["Your Amazon order has been shipped and will arrive tomorrow."],
    ["Can we reschedule our 3pm meeting to 4pm today?"],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #080c14 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(16,185,129,0.06) 0%, transparent 50%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}

@keyframes bgPulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.1); }
}

.app-header { text-align: center; padding: 2.5rem 0 2rem; }
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #10b981 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 !important; letter-spacing: -1px;
}
.app-header p { color: #64748b; font-size: 0.95rem; margin-top: 0.5rem; }
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 1rem;
}

.section-title {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; color: #475569;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}

textarea {
    background: rgba(8, 12, 20, 0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #cbd5e1 !important;
    font-size: 0.9rem !important; line-height: 1.7 !important;
    padding: 1rem !important; resize: none !important;
}
textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

button.primary {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important; border-radius: 10px !important; color: white !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important; }
button.secondary {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #94a3b8 !important;
}

.result-card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    display: flex; flex-direction: column; gap: 6px;
    transition: all 0.3s ease; animation: fadeIn 0.4s ease;
}
.result-card:hover { border-color: rgba(99,102,241,0.3); transform: translateX(4px); }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.result-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #475569; }
.result-value { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
.reply-text { font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; font-weight: 400 !important; color: #cbd5e1 !important; line-height: 1.6; }
.result-card.empty .result-value { color: #1e293b; }
.result-card.error { border-left: 4px solid #ef4444 !important; }

.how-it-works {
    margin-top: 1rem; background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.1); border-radius: 10px;
    padding: 0.8rem 1rem; color: #475569; font-size: 0.78rem; line-height: 1.6;
}
.footer { text-align: center; color: #1e293b; font-size: 0.72rem; padding: 1.5rem 0; }
"""

with gr.Blocks(title="Email Triage AI", css=CSS) as demo:

    gr.HTML("""
    <div class='app-header'>
        <div class='badge'>AI Powered</div>
        <h1>Email Triage AI</h1>
        <p>Paste your email · Get instant classification · Know what matters</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML("<div class='section-title'>Your Email</div>")
            email_input = gr.Textbox(
                label="",
                placeholder="Paste your email content here...",
                lines=14, max_lines=18,
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze Email", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=email_input, label="Quick Examples")

        with gr.Column(scale=1):
            gr.HTML("<div class='section-title'>Analysis Results</div>")
            gr.HTML("<br>")
            category_out = gr.HTML(value="<div class='result-card empty'><span class='result-label'>CATEGORY</span><span class='result-value'>—</span></div>")
            priority_out = gr.HTML(value="<div class='result-card empty'><span class='result-label'>PRIORITY</span><span class='result-value'>—</span></div>")
            response_out = gr.HTML(value="<div class='result-card empty'><span class='result-label'>SUGGESTED REPLY</span><span class='result-value'>—</span></div>")
            gr.HTML("""
            <div class='how-it-works'>
                How it works: Llama 3.1 reads your email and classifies it as
                <span style='color:#ef4444'>spam</span>, <span style='color:#10b981'>important</span>, or <span style='color:#3b82f6'>normal</span>
                then assigns priority and drafts a reply.
            </div>
            """)

    gr.HTML("<div class='footer'>Built with Gradio · Llama 3.1 · Hugging Face</div>")

    submit_btn.click(fn=classify_email, inputs=email_input, outputs=[category_out, priority_out, response_out])
    clear_btn.click(
        fn=lambda: (
            "",
            "<div class='result-card empty'><span class='result-label'>CATEGORY</span><span class='result-value'>—</span></div>",
            "<div class='result-card empty'><span class='result-label'>PRIORITY</span><span class='result-value'>—</span></div>",
            "<div class='result-card empty'><span class='result-label'>SUGGESTED REPLY</span><span class='result-value'>—</span></div>",
        ),
        outputs=[email_input, category_out, priority_out, response_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)