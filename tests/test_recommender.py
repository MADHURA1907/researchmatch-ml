
import pandas as pd
import pytest

from src.recommender import (
    load_papers,
    build_recommender,
    recommend_papers,
)


@pytest.fixture
def recommender():
    papers = load_papers()
    vectorizer, matrix = build_recommender(papers)
    return papers, vectorizer, matrix


def test_dataset_loads(recommender):
    papers, _, _ = recommender
    assert len(papers) == 12
    assert {"title", "abstract", "category", "url"}.issubset(
        papers.columns
    )


def test_relevant_papers_are_ranked(recommender):
    papers, vectorizer, matrix = recommender
    results = recommend_papers(
        "transformer language models",
        papers,
        vectorizer,
        matrix,
        top_k=3,
    )

    assert not results.empty
    assert len(results) <= 3
    assert results.iloc[0]["similarity"] > 0
    assert "Language Models are Few-Shot Learners" in (
        results.iloc[0]["title"]
    )


def test_empty_query_returns_no_results(recommender):
    papers, vectorizer, matrix = recommender
    results = recommend_papers(
        "   ", papers, vectorizer, matrix
    )
    assert results.empty


def test_top_k_must_be_positive(recommender):
    papers, vectorizer, matrix = recommender
    with pytest.raises(ValueError):
        recommend_papers(
            "computer vision",
            papers,
            vectorizer,
            matrix,
            top_k=0,
        )
