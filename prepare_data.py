"""Download selected public Hugging Face datasets and merge a clean CSV.\nInstall: pip install datasets pandas\nRun: python prepare_data.py\n"""
from datasets import load_dataset
import pandas as pd
SOURCES = [
    ("S-Labs/prompt-injection-dataset", None),
    ("deepset/prompt-injections", None),
    ("axiotic/ogma-prompt-injection-benign", "train"),
    ("guychuk/benign-malicious-prompt-classification", "train"),
]
MAX_ROWS_PER_SOURCE = 50_000

def label_value(value):
    if isinstance(value, (int, float)) and value in (0, 1): return int(value)
    s = str(value).strip().lower()
    if s in {"0", "benign", "safe", "normal", "non_injection", "non-injection"}: return 0
    if s in {"1", "injection", "malicious", "unsafe", "attack", "jailbreak"}: return 1
    raise ValueError(f"Unrecognized label: {value!r}")

def collect(repo, requested_split):
    loaded = load_dataset(repo)
    splits = [requested_split] if requested_split else list(loaded.keys())
    chunks = []
    for split in splits:
        if split not in loaded: continue
        ds = loaded[split]
        cols = ds.column_names
        text_col = next((c for c in ("text", "prompt", "input", "instruction") if c in cols), None)
        label_col = next((c for c in ("label", "labels", "class") if c in cols), None)
        if not text_col or not label_col: raise ValueError(f"{repo}/{split}: expected text and label columns; found {cols}")
        frame = ds.select(range(min(len(ds), MAX_ROWS_PER_SOURCE))).to_pandas()
        out = pd.DataFrame({"text": frame[text_col], "label": frame[label_col]})
        out["label"] = out["label"].map(label_value)
        out["source"], out["source_split"] = repo, split
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

frames = []
for repo, split in SOURCES:
    print("Loading", repo)
    frames.append(collect(repo, split))
df = pd.concat(frames, ignore_index=True)
df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df.text.str.len().between(4, 4000)]
conflict = df.groupby("text").label.nunique()
conflicting = set(conflict[conflict > 1].index)
df = df[~df.text.isin(conflicting)].drop_duplicates("text").reset_index(drop=True)
df.to_csv("prompt_injection.csv", index=False)
print(f"Saved {len(df):,} rows; removed {len(conflicting):,} label conflicts")
print(df.label.value_counts().rename(index={0:"benign",1:"injection"}))
