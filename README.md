# Prompt injection detector starter kit

## Files

- `prepare_data.py` downloads and merges four public Hugging Face datasets into `prompt_injection.csv`.
- `prompt_injection_distilbert_colab.ipynb` is the Colab fine-tuning notebook.

## Data sources

The prep script combines the MIT-licensed S-Labs dataset (15k, includes hard negatives), Apache-2.0 deepset dataset (662), Apache-2.0 Axiotic benign prompts (11k), and Apache-2.0 Guychuk binary dataset (464k, capped at 50k rows). Dataset cards and licenses can change; check the cards before redistribution. See [S-Labs](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset), [deepset](https://huggingface.co/datasets/deepset/prompt-injections), [Axiotic](https://huggingface.co/datasets/axiotic/ogma-prompt-injection-benign), and [Guychuk](https://huggingface.co/datasets/guychuk/benign-malicious-prompt-classification).

## Run

```bash
pip install datasets pandas
python prepare_data.py
```

Then upload the resulting CSV to Colab, select a GPU runtime if available, and run the notebook cells. The notebook saves `prompt-injection-distilbert.zip`.

The CSV includes `text`, `label` (0 benign, 1 injection), `source`, and `source_split`. Preparation removes empty/very long examples, exact duplicates, and any text with contradictory labels. Collection downloads data from Hugging Face; internet access is needed.
