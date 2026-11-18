"""Download public prompt-injection datasets and write a clean train/test split.

Output:
    data/train/train.csv   (text, label, category, source)
    data/test/test.csv     (held-out: NEVER train or tune on this)

label: 1 = prompt injection / jailbreak attack, 0 = genuine.
category: which public dataset the row came from (none of them ship attack categories).

Run:  python prepare_data.py
"""
import hashlib
import re
import warnings
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# (repo, short name, {original split -> our split}).
# Where a dataset has its own test split we keep it as test; datasets without one
# get a deterministic 15% hash-based hold-out ("hash").
SOURCES = [
    ("S-Labs/prompt-injection-dataset", "s-labs",
     {"train": "train", "validation": "train", "test": "test"}),
    ("deepset/prompt-injections", "deepset", {"train": "train", "test": "test"}),
    ("axiotic/ogma-prompt-injection-benign", "axiotic", {"train": "hash"}),
]
# NOT used: guychuk/benign-malicious-prompt-classification. Its label 1 mixes prompt
# injections with plain harmful requests ("How is meth made?"), and its label 0 also
# contains harmful requests, so "1" does not mean "prompt injection" there. It is
# also dominated by near-identical HackAPrompt templates.

TEST_FRACTION = 0.15
NEAR_DUP_SIMILARITY = 0.90  # cosine similarity on char n-grams
OUT_TRAIN = Path("data/train/train.csv")
OUT_TEST = Path("data/test/test.csv")


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def norm(s):
    """Normalised form for duplicate detection: lowercase, letters/digits only."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def hash_is_test(text):
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return (h % 1000) < TEST_FRACTION * 1000


def load_source(repo, name, split_map):
    ds = load_dataset(repo)
    frames = []
    for orig_split, our_split in split_map.items():
        d = ds[orig_split].to_pandas()
        out = pd.DataFrame({"text": d["text"].map(clean), "label": d["label"].astype(int)})
        out["category"] = name
        out["source"] = f"{repo}:{orig_split}"
        out["split"] = our_split
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    assert set(df.label.unique()) <= {0, 1}, f"{repo}: unexpected labels {df.label.unique()}"
    hashed = df.split == "hash"
    df.loc[hashed, "split"] = df.loc[hashed, "text"].map(lambda t: "test" if hash_is_test(t) else "train")
    return df


def main():
    df = pd.concat([load_source(*s) for s in SOURCES], ignore_index=True)
    n_raw = len(df)
    df = df[df.text.str.len().between(4, 4000)].copy()
    df["key"] = df.text.map(norm)

    # Drop texts that appear with both labels anywhere (contradictory).
    conflict = df.groupby("key").label.nunique()
    bad = set(conflict[conflict > 1].index)
    df = df[~df.key.isin(bad)]

    # If a text appears in test anywhere, it belongs to test only.
    test_keys = set(df.loc[df.split == "test", "key"])
    df.loc[df.key.isin(test_keys), "split"] = "test"
    df = df.drop_duplicates("key")

    train, test = df[df.split == "train"].copy(), df[df.split == "test"].copy()

    # Remove train rows that are near-duplicates of any test row.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit(pd.concat([train.text, test.text]))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(test.text))
    dist, _ = nn.kneighbors(vec.transform(train.text))
    near = (1 - dist[:, 0]) >= NEAR_DUP_SIMILARITY
    train = train[~near]

    cols = ["text", "label", "category", "source"]
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEST.parent.mkdir(parents=True, exist_ok=True)
    train[cols].sample(frac=1, random_state=42).to_csv(OUT_TRAIN, index=False)
    test[cols].sample(frac=1, random_state=42).to_csv(OUT_TEST, index=False)

    print(f"Raw rows: {n_raw:,}; label conflicts removed: {len(bad):,}; "
          f"near-duplicates of test removed from train: {int(near.sum()):,}")
    print(f"Wrote {OUT_TRAIN} ({len(train):,} rows) and {OUT_TEST} ({len(test):,} rows)")

    # Final overlap check, re-read from disk.
    tr, te = pd.read_csv(OUT_TRAIN), pd.read_csv(OUT_TEST)
    overlap = set(tr.text.map(norm)) & set(te.text.map(norm))
    print(f"Exact/normalised train-test overlap: {len(overlap)}")
    assert not overlap, "train/test overlap found"


if __name__ == "__main__":
    main()
