# 📧 AI Email Triage Agent

An AI-powered email assistant that can **classify emails**, **assign priority**, and **generate responses** using a Large Language Model (LLM). The system is evaluated using a simulated environment and a reward-based scoring mechanism.

---

## 🚀 Features

* 📌 Classifies emails into:

  * `spam`
  * `important`
  * `normal`

* ⚡ Assigns priority levels:

  * `low`
  * `medium`
  * `high`

* 💬 Generates short, context-aware replies

* 🎯 Evaluates performance using:

  * Environment reward system
  * Grading function (score between 0–1)

---

## 🧠 Project Architecture

```
Email Input → LLM → Action → Environment → Reward → Grader → Score
```

### Components:

* **Environment (`environment.py`)**

  * Simulates incoming emails
  * Provides reward based on agent performance

* **Models (`models.py`)**

  * Defines structured data (Observation, Action, Reward)

* **Grader (`grader.py`)**

  * Evaluates agent output

* **Inference Script (`run_inference.py`)**

  * Runs the full pipeline using an LLM

---

## 📂 Project Structure

```
project/
│
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── models.py
│   ├── grader.py
│
├── tasks.py
├── run_inference.py
├── requirements.txt
├── openenv.yaml
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone <your-repo-link>
cd project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 🔑 Setup API Key

Create a `.env` file in the root directory:

```
HF_TOKEN=your_api_key_here
```

---

## ▶️ Run the Project

```
python run_inference.py
```

---

## 🧪 Example Output

```
--- Episode 1 ---
Email: Project deadline is tomorrow

Action: category='important' priority='high' response='I will complete it today.'
Reward: 1.0
Score: 1.0
```

---

## 📊 Evaluation Metrics

* **Classification Accuracy**
* **Priority Matching**
* **Response Quality (basic length check)**

Final score ranges from **0.0 to 1.0**

---

## 🧩 Tasks

Defined in `tasks.py`:

* **Easy** → Email classification
* **Medium** → + Priority assignment
* **Hard** → + Response generation

---

## 🐳 Docker (Optional)

A `Dockerfile` is included for containerized execution.

To build and run:

```
docker build -t email-agent .
docker run email-agent
```

---

## 💡 Future Improvements

* Add real-world email datasets (e.g., Enron)
* Improve response quality scoring
* Build a web interface (Streamlit)
* Add learning/feedback loop (RL-based)

---

## 🎓 Use Case

This project demonstrates:

* AI agent design
* LLM integration
* Evaluation systems
* Simulation environments

---

## 🏁 Conclusion

This project showcases how LLMs can be used to build intelligent assistants capable of understanding, prioritizing, and responding to real-world tasks like email management.

---

## 📜 License

This project is for educational purposes.
