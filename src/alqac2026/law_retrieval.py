from __future__ import annotations

import gc
import json
import re
import unicodedata
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import LawArticle, LawEvidence


LAW_CITATION_ALIASES = {
    "47/2010/QH12": ("luật các tổ chức tín dụng",),
    "66/2014/QH13": ("luật kinh doanh bất động sản",),
    "24/2012/NĐ-CP": ("nghị định 24/2012/nđ-cp", "nghị định số 24/2012/nđ-cp"),
    "60/2014/QH13": ("luật hộ tịch",),
    "52/2010/QH12": ("luật nuôi con nuôi",),
    "26/2008/QH12": ("luật thi hành án dân sự",),
    "19/2011/NĐ-CP": ("nghị định 19/2011/nđ-cp", "nghị định số 19/2011/nđ-cp"),
    "326/2016/UBTVQH14": (
        "nghị quyết 326/2016/ubtvqh14",
        "nghị quyết số 326/2016/ubtvqh14",
        "nghị quyết số 326",
    ),
    "92/2015/QH13": ("bộ luật tố tụng dân sự",),
    "39/2009/QH12": ("luật người cao tuổi",),
    "91/2015/QH13": ("bộ luật dân sự",),
    "52/2014/QH13": ("luật hôn nhân và gia đình",),
    "45/2013/QH13": ("luật đất đai",),
    "100/2015/QH13": ("bộ luật hình sự",),
    "37/2015/NĐ-CP": ("nghị định 37/2015/nđ-cp", "nghị định số 37/2015/nđ-cp"),
    "02/2011/QH13": ("luật khiếu nại",),
    "93/2015/QH13": ("luật tố tụng hành chính",),
    "50/2014/QH13": ("luật xây dựng",),
}


def _normalize_citation_text(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", value)


def extract_law_citations(
    text: str,
    articles: list[LawArticle],
    max_citations: int = 12,
    citation_window: int = 500,
) -> list[tuple[str, int]]:
    """Extract exact corpus law/article-number citations from retrieved evidence."""
    if max_citations <= 0 or not text:
        return []
    available = {(article.law_id, article.article_number) for article in articles}
    normalized = _normalize_citation_text(text)
    mentions: list[tuple[int, int, str]] = []
    for law_id, aliases in LAW_CITATION_ALIASES.items():
        normalized_aliases = {
            _normalize_citation_text(alias) for alias in (*aliases, law_id)
        }
        for alias in normalized_aliases:
            expected_year = law_id.split("/")[1]
            for match in re.finditer(re.escape(alias), normalized):
                suffix = normalized[match.end() : match.end() + 24]
                stated_year = re.search(r"\b(?:nam\s+)?((?:19|20)\d{2})\b", suffix)
                if stated_year and stated_year.group(1) != expected_year:
                    continue
                mentions.append((match.start(), match.end(), law_id))
    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    non_overlapping: list[tuple[int, int, str]] = []
    for mention in mentions:
        if non_overlapping and mention[0] < non_overlapping[-1][1]:
            continue
        non_overlapping.append(mention)

    citations: list[tuple[str, int]] = []
    previous_end = 0
    for start, end, law_id in non_overlapping:
        segment = normalized[max(previous_end, start - citation_window) : start]
        for value in re.findall(r"\bdieu\s+(\d+)", segment):
            key = (law_id, int(value))
            if key in available and key not in citations:
                citations.append(key)
                if len(citations) >= max_citations:
                    return citations
        previous_end = end
    return citations


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
        citation_k: int = 0,
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
        self.citation_k = citation_k
        self.bm25 = BM25LawRetriever(articles)
        self._article_index_by_citation = {
            (article.law_id, article.article_number): index
            for index, article in enumerate(articles)
        }
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
        citation_indices = [
            self._article_index_by_citation[key]
            for key in extract_law_citations(
                query, self.articles, max_citations=self.citation_k
            )
        ]
        indices = list(
            dict.fromkeys(citation_indices + [index for index, _ in fused])
        )
        reranker = self._load_reranker()
        pairs = [(query, self.articles[index].text) for index in indices]
        scores = reranker.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
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
        citation_k=int(config.get("citation_k", 0)),
    )
