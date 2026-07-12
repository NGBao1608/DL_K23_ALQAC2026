#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alqac2026.config import load_config
from alqac2026.data import load_law_corpus
from alqac2026.law_retrieval import HybridLawRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense law embedding index")
    parser.add_argument("--config", default="configs/candidate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["law_retrieval"]
    retriever = HybridLawRetriever(
        articles=load_law_corpus(config["paths"]["corpus"]),
        embedding_model=settings["embedding_model"],
        reranker_model=settings["reranker_model"],
        embedding_revision=settings.get("embedding_revision"),
        reranker_revision=settings.get("reranker_revision"),
        index_dir=config["paths"]["law_index"],
        sparse_k=settings["sparse_k"],
        dense_k=settings["dense_k"],
        candidate_k=settings["candidate_k"],
        top_k=settings["top_k"],
        rrf_k=settings["rrf_k"],
        batch_size=settings["batch_size"],
    )
    embeddings = retriever.build_or_load_index()
    print(f"Law index ready: {embeddings.shape}")


if __name__ == "__main__":
    main()
