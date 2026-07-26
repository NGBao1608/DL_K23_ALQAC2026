from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import LawArticle, LawEvidence


def tokenize_vietnamese(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text).split()
    except ImportError:
        return re.findall(r"\w+", text, flags=re.UNICODE)


class LawRetriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[LawEvidence]: ...


class BM25LawRetriever:
    def __init__(self, articles: list[LawArticle]):
        self.articles = articles
        self._index = BM25Okapi([tokenize_vietnamese(item.text) for item in articles])

    def ranked(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self._index.get_scores(tokenize_vietnamese(query))
        order = np.argsort(-scores)[:top_k]
        return [(int(index), float(scores[index])) for index in order]

    def search(self, query: str, top_k: int = 5) -> list[LawEvidence]:
        return [
            _as_evidence(self.articles[index], score)
            for index, score in self.ranked(query, top_k)
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], rrf_k: int = 60
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, identifier in enumerate(ranked, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _as_evidence(article: LawArticle, score: float) -> LawEvidence:
    return LawEvidence(
        law_id=article.law_id,
        aid=article.aid,
        article_number=article.article_number,
        text=article.text,
        score=float(score),
    )


class HybridLawRetriever:
    def __init__(
        self,
        articles: list[LawArticle],
        embedding_model: str,
        reranker_model: str,
        embedding_revision: str | None,
        reranker_revision: str | None,
        index_dir: str | Path,
        sparse_k: int = 50,
        dense_k: int = 50,
        candidate_k: int = 30,
        top_k: int = 5,
        rrf_k: int = 60,
        batch_size: int = 16,
    ):
        self.articles = articles
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedding_revision = embedding_revision
        self.reranker_revision = reranker_revision
        self.index_dir = Path(index_dir)
        self.sparse_k = sparse_k
        self.dense_k = dense_k
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.batch_size = batch_size
        self.bm25 = BM25LawRetriever(articles)
        self._embedding_model = None
        self._reranker = None
        self._embeddings: np.ndarray | None = None

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, revision=self.embedding_revision
            )
            self._embedding_model.max_seq_length = 2048
        return self._embedding_model

    def _index_metadata(self) -> dict:
        return {
            "embedding_model": self.embedding_model_name,
            "embedding_revision": self.embedding_revision,
            "article_keys": [article.key for article in self.articles],
        }

    def build_or_load_index(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings
        embeddings_path = self.index_dir / "law_embeddings.npy"
        metadata_path = self.index_dir / "law_embeddings.json"
        expected = self._index_metadata()
        if embeddings_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == expected:
                self._embeddings = np.load(embeddings_path, mmap_mode="r")
                return self._embeddings

        model = self._load_embedding_model()
        values = model.encode(
            [article.text for article in self.articles],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, values)
        metadata_path.write_text(
            json.dumps(expected, ensure_ascii=False), encoding="utf-8"
        )
        self._embeddings = values
        return self._embeddings

    def _dense_ranked(self, query: str) -> list[tuple[int, float]]:
        model = self._load_embedding_model()
        query_vector = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype("float32")
        scores = np.asarray(self.build_or_load_index()) @ query_vector
        order = np.argsort(-scores)[: self.dense_k]
        return [(int(index), float(scores[index])) for index in order]

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self.reranker_model_name,
                max_length=2304,
                revision=self.reranker_revision,
            )
        return self._reranker

    def search(self, query: str, top_k: int | None = None) -> list[LawEvidence]:
        final_k = top_k or self.top_k
        sparse = self.bm25.ranked(query, self.sparse_k)
        dense = self._dense_ranked(query)
        fused = reciprocal_rank_fusion(
            [[index for index, _ in sparse], [index for index, _ in dense]],
            rrf_k=self.rrf_k,
        )[: self.candidate_k]
        indices = [index for index, _ in fused]
        reranker = self._load_reranker()
        pairs = [(query, self.articles[index].text) for index in indices]
        scores = reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            zip(indices, np.asarray(scores).reshape(-1).tolist()),
            key=lambda item: -item[1],
        )[:final_k]
        return [_as_evidence(self.articles[index], score) for index, score in ranked]

    def release_gpu_models(self) -> None:
        self._embedding_model = None
        self._reranker = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_law_retriever(
    articles: list[LawArticle], config: dict
) -> BM25LawRetriever | HybridLawRetriever:
    if config["strategy"] == "bm25_only":
        return BM25LawRetriever(articles)
    if config["strategy"] != "hybrid_rerank":
        raise ValueError(f"Unsupported law retrieval strategy: {config['strategy']}")
    return HybridLawRetriever(
        articles=articles,
        embedding_model=config["embedding_model"],
        reranker_model=config["reranker_model"],
        embedding_revision=config.get("embedding_revision"),
        reranker_revision=config.get("reranker_revision"),
        index_dir=config.get("index_dir", "cache/law_index"),
        sparse_k=int(config["sparse_k"]),
        dense_k=int(config["dense_k"]),
        candidate_k=int(config["candidate_k"]),
        top_k=int(config["top_k"]),
        rrf_k=int(config["rrf_k"]),
        batch_size=int(config.get("batch_size", 16)),
    )
