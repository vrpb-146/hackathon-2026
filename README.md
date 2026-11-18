# Prompt Shield

An AI-based detector that catches prompt injection and jailbreak attempts in user messages and blocks them before they reach a chatbot, while letting genuine questions through. Built for a college hackathon.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put any API keys in a `.env` file in the project root (never commit it).

## Data

CSV files with columns `text, label, category` (`label`: 1 = attack, 0 = genuine).

- `data/train/` – training data
- `data/test/` – held-out evaluation set (never used for training)
