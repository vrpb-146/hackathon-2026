"""Print dataset stats for data/train and data/test.

Run:  python data_stats.py
"""
import re
from pathlib import Path

import pandas as pd


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load(folder):
    files = sorted(Path(folder).glob("*.csv"))
    return pd.concat([pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True)


train, test = load("data/train"), load("data/test")
df = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
df["label_name"] = df.label.map({0: "genuine", 1: "attack"})

print("Rows per dataset (category) and split:")
print(pd.crosstab([df.category, df.split], df.label_name, margins=True).to_string(), "\n")

print("Attack vs genuine per split:")
print(pd.crosstab(df.split, df.label_name, margins=True).to_string(), "\n")

for name in ("attack", "genuine"):
    print(f"5 random {name} examples (train):")
    for _, r in train[train.label == (name == "attack")].sample(5, random_state=7).iterrows():
        print(f"  [{r.category}] {r.text[:150]}")
    print()

overlap = set(train.text.map(norm)) & set(test.text.map(norm))
print(f"Train/test overlap (normalised exact match): {len(overlap)}")
