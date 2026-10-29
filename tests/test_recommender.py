"""Unit tests for ResearchMatch recommendation engine."""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.recommender import (
    DEFAULT_RRF_K,
    build_recommender,
    build_semantic_embeddings,
    get_related_papers,
    get_semantic_model,
    load_papers,
    normalize_text,
    recommend_hybrid,
    recommend_papers,
    recommend_semantic,
)


@pytest.fixture
def recommender():
    papers = load_papers()
    vectorizer, matrix = build_recommender(papers)
    return papers, vectorizer, matrix


# ---------------------------------------------------------------------------
# Existing Tests Preserved
# ---------------------------------------------------------------------------

def test_dataset_loads(recommender):
    papers, _, _ = recommender
    assert len(papers) == 12
    assert {"title", "abstract", "category", "url"}.issubset(papers.columns)


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
    assert "Language Models are Few-Shot Learners" in results.iloc[0]["title"]


def test_empty_query_returns_no_results(recommender):
    papers, vectorizer, matrix = recommender
    results = recommend_papers("   ", papers, vectorizer, matrix)
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


def test_recommendations_have_valid_similarity_scores(recommender):
    papers, vectorizer, matrix = recommender
    results = recommend_papers(
        "computer vision",
        papers,
        vectorizer,
        matrix,
        top_k=5,
    )

    assert not results.empty
    assert results["similarity"].between(0, 1).all()
    assert results["similarity"].is_monotonic_decreasing


def test_top_k_limits_number_of_results(recommender):
    papers, vectorizer, matrix = recommender
    results = recommend_papers(
        "generative AI",
        papers,
        vectorizer,
        matrix,
        top_k=2,
    )

    assert len(results) <= 2


def test_invalid_dataset_path_raises_error():
    with pytest.raises((FileNotFoundError, OSError)):
        load_papers("data/does_not_exist.csv")


# ---------------------------------------------------------------------------
# New Tests: Preprocessing & Data Validation
# ---------------------------------------------------------------------------

def test_normalize_text_helper():
    assert normalize_text("   hello    world  ") == "hello world"
    assert normalize_text(None) == ""
    assert normalize_text(np.nan) == ""


def test_load_papers_deduplication(tmp_path):
    csv_file = tmp_path / "dup_papers.csv"
    csv_file.write_text(
        "id,title,abstract,category,year,url\n"
        "1,Attention Is All You Need,Original paper,NLP,2017,http://a.com\n"
        "2,attention is all you need!,Duplicate title,NLP,2017,http://b.com\n"
        "1,Different Paper,Same ID,NLP,2018,http://c.com\n"
        "3,BERT: Pre-training,Pre-training BERT,NLP,2018,http://d.com\n"
    )

    loaded = load_papers(csv_file)
    assert len(loaded) == 2
    assert list(loaded["id"]) == [1, 3]


def test_load_papers_missing_required_column(tmp_path):
    csv_file = tmp_path / "bad_columns.csv"
    csv_file.write_text(
        "id,title,category,year,url\n"
        "1,Test Paper,NLP,2020,http://example.com\n"
    )
    with pytest.raises(ValueError, match="Dataset is missing columns"):
        load_papers(csv_file)


# ---------------------------------------------------------------------------
# New Tests: Semantic Search (Mocked for speed and deterministic testing)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_env(recommender):
    papers, _, _ = recommender
    n = len(papers)
    dim = 8

    # Create distinct synthetic normalized vectors
    rng = np.random.default_rng(42)
    fake_embeddings = rng.standard_normal((n, dim))
    fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

    mock_model = MagicMock()
    # Mock query encoding to match paper index 0 closely
    mock_model.encode.side_effect = lambda docs, **kwargs: (
        fake_embeddings[:1] if len(docs) == 1 else fake_embeddings
    )

    return papers, fake_embeddings, mock_model


def test_recommend_semantic_with_mock(mock_embedding_env):
    papers, embeddings, mock_model = mock_embedding_env
    results = recommend_semantic(
        "transformer models",
        papers,
        embeddings,
        model=mock_model,
        top_k=4,
    )

    assert not results.empty
    assert len(results) <= 4
    assert "similarity" in results.columns
    assert results["similarity"].between(0, 1).all()
    assert results["similarity"].is_monotonic_decreasing


def test_recommend_semantic_empty_query(mock_embedding_env):
    papers, embeddings, mock_model = mock_embedding_env
    results = recommend_semantic(
        "   ",
        papers,
        embeddings,
        model=mock_model,
        top_k=3,
    )
    assert results.empty


def test_recommend_semantic_dimension_mismatch(mock_embedding_env):
    papers, embeddings, mock_model = mock_embedding_env
    with pytest.raises(ValueError, match="Mismatch between papers"):
        recommend_semantic(
            "test query",
            papers.iloc[:5],
            embeddings,
            model=mock_model,
        )


# ---------------------------------------------------------------------------
# New Tests: Hybrid Retrieval (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def test_recommend_hybrid_fuses_results(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    hybrid_results = recommend_hybrid(
        query="transformer language models",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=5,
        rrf_k=60,
    )

    assert not hybrid_results.empty
    assert len(hybrid_results) <= 5
    assert "rrf_score" in hybrid_results.columns
    assert "lexical_rank" in hybrid_results.columns
    assert "semantic_rank" in hybrid_results.columns
    assert "lexical_score" in hybrid_results.columns
    assert "semantic_score" in hybrid_results.columns
    # Check monotonic decreasing by RRF score
    assert hybrid_results["rrf_score"].is_monotonic_decreasing


def test_recommend_hybrid_empty_query(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    results = recommend_hybrid(
        query="   ",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
    )
    assert results.empty


# ---------------------------------------------------------------------------
# New Tests: Related Paper Recommendations (Item-to-Item)
# ---------------------------------------------------------------------------

def test_get_related_papers_excludes_self(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, _ = mock_embedding_env

    target_id = papers.iloc[0]["id"]
    related = get_related_papers(
        paper_id=target_id,
        papers=papers,
        embeddings=embeddings,
        top_k=3,
        method="semantic",
    )

    assert not related.empty
    assert len(related) <= 3
    # Target paper must not be in its own related recommendations
    assert target_id not in related["id"].values
    assert related["similarity"].is_monotonic_decreasing


def test_get_related_papers_tfidf_fallback(recommender):
    papers, vectorizer, tfidf_matrix = recommender
    target_id = papers.iloc[1]["id"]

    related = get_related_papers(
        paper_id=target_id,
        papers=papers,
        vectorizer=vectorizer,
        matrix=tfidf_matrix,
        top_k=3,
        method="tfidf",
    )

    assert not related.empty
    assert target_id not in related["id"].values


def test_get_related_papers_invalid_id_raises_error(recommender):
    papers, vectorizer, tfidf_matrix = recommender
    with pytest.raises(ValueError, match="not found in dataset"):
        get_related_papers(
            paper_id=999999,
            papers=papers,
            vectorizer=vectorizer,
            matrix=tfidf_matrix,
        )


# ---------------------------------------------------------------------------
# Real Semantic Model Smoke Test
# ---------------------------------------------------------------------------

def test_real_semantic_model_smoke(recommender):
    """Smoke test running actual sentence-transformers embedding inference."""
    papers, _, _ = recommender
    model = get_semantic_model()
    assert model is not None

    embeddings = build_semantic_embeddings(papers, model=model)
    assert embeddings.shape == (len(papers), 384)

    results = recommend_semantic(
        "attention mechanism for sequence modeling",
        papers,
        embeddings,
        model=model,
        top_k=3,
    )
    assert not results.empty
    # Attention Is All You Need should be top-ranked for this query
    assert "Attention Is All You Need" in results.iloc[0]["title"]
