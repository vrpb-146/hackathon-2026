# Prompt Shield

An AI-based detector that catches prompt injection and jailbreak attempts in user messages and blocks them before they reach a chatbot, while letting genuine questions through. Built for a college hackathon.

## Setup

Use Python 3.12 (3.15 is too new: several packages have no prebuilt wheels yet).

```powershell
py install 3.12
py -V:3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put any API keys in a `.env` file in the project root (never commit it).

## Files

- `prepare_data.py` downloads public Hugging Face datasets and writes `data/train/train.csv` and the held-out `data/test/test.csv` (no overlap, near-duplicates removed).
- `data_stats.py` prints rows per dataset, attack vs genuine counts, examples, and a train/test overlap check.
- `prompt_injection_distilbert_colab.ipynb` fine-tunes DistilBERT in Colab on `train.csv` and evaluates once on `test.csv`.

## Data

CSV columns: `text, label, category` (+ `source`). `label`: 1 = prompt injection / jailbreak, 0 = genuine.

- `data/train/` – training data
- `data/test/` – held-out evaluation set (never used for training or tuning)

Sources: [S-Labs](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) (MIT), [deepset](https://huggingface.co/datasets/deepset/prompt-injections) (Apache-2.0), [Axiotic benign](https://huggingface.co/datasets/axiotic/ogma-prompt-injection-benign) (Apache-2.0). Check the dataset cards before redistribution.

`guychuk/benign-malicious-prompt-classification` was deliberately left out: its "malicious" label mixes prompt injections with plain harmful requests, and its "benign" label also contains harmful requests.
