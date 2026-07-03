"""
End-to-end runner: produce predictions for a test set and write a submission file.

Usage:
  # Offline self-evaluation on the labeled public test (no leaderboard needed):
  python src/run_predict.py --split public --eval

  # Generate a private-test submission file (then upload via src/submit.py):
  python src/run_predict.py --split private --in data/raw/private_test.json \
         --out submissions/run1.json

The output format is a list of {"case_id", "label"} — ADAPT to the exact format
required by the api-docs page before submitting.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from data_utils import load_corpus, load_public_test, load_private_test, LABELS, Case
from retrieval import BM25Retriever  # add DenseRetriever / rrf_fuse / rerank when GPU available
from agent import VerdictAgent, build_few_shots


def get_retrieved_texts(retriever, id2text, query, k=5):
    hits = retriever.search(query, k=k)
    return [(doc_id, id2text[doc_id]) for doc_id, _ in hits]


def run(cases: list[Case], retriever, id2text, agent, few_shots, k=5, sleep=0.0):
    preds = []
    for i, c in enumerate(cases, 1):
        retrieved = get_retrieved_texts(retriever, id2text, c.case_query, k=k)
        out = agent.predict(c.case_query, retrieved, few_shots)
        preds.append({"case_id": c.case_id, "label": out["label"]})
        if i % 5 == 0:
            print(f"  {i}/{len(cases)} done")
        if sleep:
            time.sleep(sleep)
    return preds


def evaluate(preds, cases):
    from sklearn.metrics import f1_score, classification_report
    gold = {c.case_id: c.verdict_label for c in cases}
    y_true = [gold[p["case_id"]] for p in preds]
    y_pred = [p["label"] for p in preds]
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\nAccuracy={acc:.3f}  Macro-F1={macro:.3f}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    return macro


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default="submissions/run.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the LLM; use majority-class to test the plumbing")
    args = ap.parse_args()

    arts = load_corpus(os.getenv("CORPUS_PATH", "data/raw/corpus_law_pub.json"))
    id2text = {a.doc_id: a.text for a in arts}

    public = load_public_test(os.getenv("PUBLIC_TEST_PATH", "data/raw/ALQAC2026_public_test.json"))
    few_shots = build_few_shots(public, n_per_class=1)

    if args.split == "public":
        cases = public
    else:
        cases = load_private_test(args.inp)

    print("Building retriever (BM25)...")
    retriever = BM25Retriever(arts)

    if args.dry_run:
        preds = [{"case_id": c.case_id, "label": "PARTIAL_A_WIN"} for c in cases]
    else:
        agent = VerdictAgent(model_name=args.model)
        preds = run(cases, retriever, id2text, agent, few_shots, k=args.k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(preds)} predictions -> {args.out}")

    if args.eval and args.split == "public":
        evaluate(preds, public)


if __name__ == "__main__":
    main()
