"""Evaluate the detector on the held-out test set and explain the results in plain English.

Run:  python evaluate.py [path/to/test.csv]
"""
import sys
import time

import pandas as pd

from shield import THRESHOLD, attack_scores

path = sys.argv[1] if len(sys.argv) > 1 else "data/test/test.csv"
df = pd.read_csv(path)

attack_scores(["warm-up"])  # load the model before timing
start = time.perf_counter()
df["score"] = attack_scores(df.text.tolist())
ms_per_msg = (time.perf_counter() - start) / len(df) * 1000
df["blocked"] = df.score >= THRESHOLD

attacks, genuine = df[df.label == 1], df[df.label == 0]
tp = int(attacks.blocked.sum()); fn = len(attacks) - tp
fp = int(genuine.blocked.sum()); tn = len(genuine) - fp
blocked_total = tp + fp
precision = tp / blocked_total if blocked_total else 0

print(f"Test set: {len(df):,} messages ({len(attacks):,} attacks, {len(genuine):,} genuine). Threshold {THRESHOLD}.\n")
print(f"Attacks blocked:         {tp:,} of {len(attacks):,} ({tp / len(attacks):.1%})")
print(f"Attacks that got through: {fn:,} ({fn / len(attacks):.1%})")
print(f"Genuine allowed:         {tn:,} of {len(genuine):,} ({tn / len(genuine):.1%})")
print(f"Genuine wrongly blocked: {fp:,} ({fp / len(genuine):.1%})")
print(f"When it blocks, it is right {precision:.1%} of the time.")
print(f"Overall accuracy:        {(tp + tn) / len(df):.1%}")
print(f"Speed on this CPU:       {ms_per_msg:.0f} ms per message (batched)\n")

print("By dataset:")
for cat, g in df.groupby("category"):
    a, n = g[g.label == 1], g[g.label == 0]
    parts = []
    if len(a):
        parts.append(f"blocked {a.blocked.sum()}/{len(a)} attacks ({a.blocked.mean():.1%})")
    if len(n):
        parts.append(f"wrongly blocked {n.blocked.sum()}/{len(n)} genuine ({n.blocked.mean():.1%})")
    print(f"  {cat:8} " + ", ".join(parts))

pd.set_option("display.max_colwidth", 140)
print("\nAttacks that got through (lowest scores first):")
for _, r in attacks[~attacks.blocked].nsmallest(8, "score").iterrows():
    print(f"  {r.score:.3f} [{r.category}] {r.text[:140]}")
print("\nGenuine messages wrongly blocked (highest scores first):")
for _, r in genuine[genuine.blocked].nlargest(8, "score").iterrows():
    print(f"  {r.score:.3f} [{r.category}] {r.text[:140]}")
