# Prompt Shield

College hackathon project.

## Problem
Build an AI-based detector that identifies prompt injection / jailbreak attempts in user messages and blocks them before they reach a chatbot, while letting genuine questions through. We must show **measurable results** of how much the detector improves the chatbot's security (e.g. attack block rate, false-positive rate on genuine questions, with vs. without the detector).

## Data
- Teammates upload CSVs to GitHub with columns: `text, label, category`.
- `label`: 1 = attack, 0 = genuine.
- `data/test/` is a **held-out set — never train on it**, never use it for tuning. Only for final evaluation.

## Rules
- Python + Streamlit.
- Keep it simple.
- API keys go only in a `.env` file (loaded with python-dotenv), never in code. The repo is **public**.
- Never commit large model files (`models/`, `*.pkl`, `*.joblib`, `*.pt`, etc. are gitignored).
