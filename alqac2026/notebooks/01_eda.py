"""
01 — Exploratory Data Analysis for ALQAC 2026.
Run: python notebooks/01_eda.py   (from repo root)
"""
import os, sys, json
from collections import Counter
sys.path.insert(0, "src")
from dotenv import load_dotenv
from data_utils import load_corpus, load_public_test, LABELS

load_dotenv()
arts = load_corpus(os.getenv("CORPUS_PATH", "data/raw/corpus_law_pub.json"))
cases = load_public_test(os.getenv("PUBLIC_TEST_PATH", "data/raw/ALQAC2026_public_test.json"))

print("=" * 60)
print(f"Corpus: {len(arts)} articles across {len(set(a.law_id for a in arts))} laws")
print("Articles per law:")
for lid, n in Counter(a.law_id for a in arts).most_common():
    print(f"  {lid:<22} {n}")

print("=" * 60)
print(f"Public test: {len(cases)} cases")
print("Label distribution:", dict(Counter(c.verdict_label for c in cases)))

# query length stats
qlens = sorted(len(c.case_query) for c in cases)
print(f"case_query length: min={qlens[0]} median={qlens[len(qlens)//2]} max={qlens[-1]}")

# gold-link stats
n_gold = [len(c.gold_law_refs) for c in cases]
print(f"resolved gold links per case: avg={sum(n_gold)/len(n_gold):.1f} max={max(n_gold)}")

# which laws are most cited as gold
cited = Counter()
for c in cases:
    for ref in c.gold_law_refs:
        cited[ref.split("|")[0]] += 1
print("Most-cited laws in gold links:", cited.most_common(6))

# class vs query-length sanity (do losers have longer queries? just curious)
print("=" * 60)
by_label = {l: [] for l in LABELS}
for c in cases:
    if c.verdict_label in by_label:
        by_label[c.verdict_label].append(len(c.case_query))
for l in LABELS:
    xs = by_label[l]
    if xs:
        print(f"  {l:<14} n={len(xs):<3} mean_query_len={sum(xs)/len(xs):.0f}")
