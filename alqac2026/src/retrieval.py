"""
Retrieval for ALQAC 2026: find the law articles relevant to a case query.

Pipeline options (combine as you like):
  1. Sparse  : BM25 over word-segmented articles.
  2. Dense   : sentence-transformers embedding + FAISS (recommended:
               'AITeamVN/Vietnamese_Embedding' = BGE-M3 fine-tuned for VN legal,
               or 'bkai-foundation-models/vietnamese-bi-encoder').
  3. Hybrid  : Reciprocal Rank Fusion (RRF) of sparse + dense, then a
               cross-encoder reranker ('AITeamVN/Vietnamese_Reranker' / 'BAAI/bge-reranker-v2-m3').

Important preprocessing: the case_query is full of party names ("Chị Lê Thị T",
"anh Chử Lương D") which pollute lexical matching. We mask them before BM25.
All chosen models are open-weight and well under 10B params (rule-compliant).
"""
from __future__ import annotations
import re
import pickle
from pathlib import Path

from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

from data_utils import Article

# Party-name patterns: "<title> <Name...> <SingleCapLetter>" e.g. "chị Lê Thị T"
_TITLES = r"(?:ông|bà|anh|chị|em|cháu|bị đơn|nguyên đơn|người|công ty|ông\.|bà\.)"
_NAME_RE = re.compile(rf"\b{_TITLES}\s+[A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){{0,3}}\b", re.IGNORECASE)


def mask_parties(text: str) -> str:
    """Replace party-name spans with a neutral token to reduce lexical noise."""
    return _NAME_RE.sub(" [BEN] ", text)


def vi_tokenize(text: str, mask: bool = True) -> list[str]:
    if mask:
        text = mask_parties(text)
    text = re.sub(r"\s+", " ", text)
    return ViTokenizer.tokenize(text.lower()).split()


# --------------------------------------------------------------------------- #
# Sparse: BM25
# --------------------------------------------------------------------------- #
class BM25Retriever:
    def __init__(self, articles: list[Article], max_chars: int = 2000):
        self.articles = articles
        self.doc_ids = [a.doc_id for a in articles]
        self._tok = [vi_tokenize(a.text[:max_chars], mask=False) for a in articles]
        self.bm25 = BM25Okapi(self._tok)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(vi_tokenize(query, mask=True))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


# --------------------------------------------------------------------------- #
# Dense: sentence-transformers + FAISS   (requires GPU/CPU + HF download)
# --------------------------------------------------------------------------- #
class DenseRetriever:
    def __init__(self, articles: list[Article],
                 model_name: str = "AITeamVN/Vietnamese_Embedding",
                 max_seq_length: int = 2048, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        self.articles = articles
        self.doc_ids = [a.doc_id for a in articles]
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length

        emb = self.model.encode([a.text for a in articles],
                                batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        import numpy as np
        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(self.doc_ids[i], float(s)) for i, s in zip(I[0], D[0])]


# --------------------------------------------------------------------------- #
# Hybrid fusion + rerank
# --------------------------------------------------------------------------- #
def rrf_fuse(*result_lists: list[tuple[str, float]], k: int = 20, c: int = 60):
    """Reciprocal Rank Fusion of several ranked lists -> top-k doc_ids."""
    score: dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(score.items(), key=lambda x: -x[1])[:k]


def rerank(query: str, doc_ids: list[str], id2text: dict[str, str],
           model_name: str = "AITeamVN/Vietnamese_Reranker", top_k: int = 5):
    """Cross-encoder rerank of candidate articles (open-weight)."""
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, max_length=2048)
    pairs = [(query, id2text[d]) for d in doc_ids]
    scores = ce.predict(pairs)
    ranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return ranked[:top_k]


# --------------------------------------------------------------------------- #
# Evaluation: Recall@k and Macro-F2 against gold law links
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    hit = len(set(retrieved[:k]) & set(gold))
    return hit / len(set(gold))


def f2_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    pred, g = set(retrieved[:k]), set(gold)
    tp = len(pred & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(g)
    beta2 = 4.0
    return (1 + beta2) * prec * rec / (beta2 * prec + rec)


if __name__ == "__main__":
    import os
    from statistics import fmean
    from dotenv import load_dotenv
    from data_utils import load_corpus, load_public_test
    load_dotenv()

    arts = load_corpus(os.getenv("CORPUS_PATH", "data/raw/corpus_law_pub.json"))
    cases = load_public_test(os.getenv("PUBLIC_TEST_PATH", "data/raw/ALQAC2026_public_test.json"))

    print("Building BM25 index...")
    bm = BM25Retriever(arts)

    for K in (5, 10, 20):
        recs, f2s = [], []
        for c in cases:
            if not c.gold_law_refs:
                continue
            got = [d for d, _ in bm.search(c.case_query, k=K)]
            recs.append(recall_at_k(got, c.gold_law_refs, K))
            f2s.append(f2_at_k(got, c.gold_law_refs, K))
        print(f"BM25  Recall@{K}={fmean(recs):.3f}  F2@{K}={fmean(f2s):.3f}  (n={len(recs)})")
