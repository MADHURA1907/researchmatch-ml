"""Unit tests for ResearchMatch recommendation engine."""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.recommender import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_RRF_K,
    build_recommender,
    build_semantic_embeddings,
    compute_corpus_hash,
    escape_bibtex,
    export_to_bibtex,
    get_related_papers,
    get_semantic_model,
    load_or_compute_embeddings,
    load_papers,
    normalize_text,
    recommend_hybrid,
    recommend_papers,
    recommend_semantic,
    save_embeddings,
    search_papers,
    validate_embeddings,
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
# Preprocessing & Data Validation
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
# Semantic Search (Mocked for speed and deterministic testing)
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
# Hybrid Retrieval (Reciprocal Rank Fusion)
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


def test_recommend_hybrid_candidate_pooling_equivalence(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    # Full corpus unpooled
    res_unpooled = recommend_hybrid(
        query="transformer language models",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=4,
        candidate_pool_size=None,
    )

    # Explicit pool size equal to corpus size
    res_pooled = recommend_hybrid(
        query="transformer language models",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=4,
        candidate_pool_size=len(papers),
    )

    assert not res_unpooled.empty
    assert len(res_unpooled) == len(res_pooled)
    assert list(res_unpooled["id"]) == list(res_pooled["id"])
    assert np.allclose(res_unpooled["rrf_score"], res_pooled["rrf_score"])


def test_recommend_hybrid_candidate_pool_size_clamping(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    # When candidate_pool_size < top_k, pool depth is clamped to at least top_k
    results = recommend_hybrid(
        query="transformer models",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=5,
        candidate_pool_size=2,
    )
    assert not results.empty
    assert len(results) <= 5

    # When candidate_pool_size <= 0, raises ValueError
    with pytest.raises(ValueError, match="candidate_pool_size must be at least 1"):
        recommend_hybrid(
            query="test",
            papers=papers,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=embeddings,
            semantic_model=mock_model,
            candidate_pool_size=0,
        )


def test_recommend_hybrid_candidate_pooling_limits_candidates(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    with patch("src.recommender.recommend_papers", wraps=recommend_papers) as mock_lex, \
         patch("src.recommender.recommend_semantic", wraps=recommend_semantic) as mock_sem:
        recommend_hybrid(
            query="transformer",
            papers=papers,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=embeddings,
            semantic_model=mock_model,
            top_k=2,
            candidate_pool_size=4,
        )
        assert mock_lex.call_count == 1
        assert mock_sem.call_count == 1
        assert mock_lex.call_args[1]["top_k"] == 4
        assert mock_sem.call_args[1]["top_k"] == 4


def test_recommend_hybrid_pure_lexical_weight(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    lexical_only = recommend_hybrid(
        query="transformer",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=3,
        lexical_weight=1.0,
        semantic_weight=0.0,
    )
    lexical_baseline = recommend_papers(
        query="transformer",
        papers=papers,
        vectorizer=vectorizer,
        matrix=tfidf_matrix,
        top_k=3,
    )
    assert not lexical_only.empty
    assert lexical_only.iloc[0]["id"] == lexical_baseline.iloc[0]["id"]


def test_recommend_hybrid_pure_semantic_weight(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    semantic_only = recommend_hybrid(
        query="transformer",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=3,
        lexical_weight=0.0,
        semantic_weight=1.0,
    )
    semantic_baseline = recommend_semantic(
        query="transformer",
        papers=papers,
        embeddings=embeddings,
        model=mock_model,
        top_k=3,
    )
    assert not semantic_only.empty
    assert semantic_only.iloc[0]["id"] == semantic_baseline.iloc[0]["id"]


def test_recommend_hybrid_invalid_weights_or_k(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    # Negative weights
    with pytest.raises(ValueError, match="must be non-negative"):
        recommend_hybrid("q", papers, vectorizer, tfidf_matrix, embeddings, lexical_weight=-1.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        recommend_hybrid("q", papers, vectorizer, tfidf_matrix, embeddings, semantic_weight=-0.5)

    # Both weights zero
    with pytest.raises(ValueError, match="At least one.*must be positive"):
        recommend_hybrid("q", papers, vectorizer, tfidf_matrix, embeddings, lexical_weight=0.0, semantic_weight=0.0)

    # Invalid rrf_k
    with pytest.raises(ValueError, match="rrf_k must be at least 1"):
        recommend_hybrid("q", papers, vectorizer, tfidf_matrix, embeddings, rrf_k=0)


def test_recommend_hybrid_tie_breaking_determinism():
    papers = pd.DataFrame([
        {
            "id": 99,
            "title": "Identical Paper Title",
            "abstract": "Identical abstract for testing tie breaks.",
            "category": "Computer Science",
            "year": 2024,
            "url": "https://example.com/99",
        },
        {
            "id": 42,
            "title": "Identical Paper Title",
            "abstract": "Identical abstract for testing tie breaks.",
            "category": "Computer Science",
            "year": 2024,
            "url": "https://example.com/42",
        },
    ])
    vectorizer, matrix = build_recommender(papers)
    embeddings = np.ones((2, 8), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((1, 8), dtype=np.float32)

    # Run 1: original order (id 99 first in df, id 42 second)
    res1 = recommend_hybrid(
        query="identical abstract",
        papers=papers,
        vectorizer=vectorizer,
        tfidf_matrix=matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=2,
    )
    # ID 42 should be ranked first because of ascending ID tie-breaker
    assert list(res1["id"]) == [42, 99]

    # Run 2: reversed DataFrame order (id 42 first in df, id 99 second)
    papers_rev = papers.iloc[::-1].reset_index(drop=True)
    matrix_rev = matrix[::-1]
    res2 = recommend_hybrid(
        query="identical abstract",
        papers=papers_rev,
        vectorizer=vectorizer,
        tfidf_matrix=matrix_rev,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=2,
    )
    assert list(res2["id"]) == [42, 99]


# ---------------------------------------------------------------------------
# Unified Search Dispatcher (search_papers)
# ---------------------------------------------------------------------------

def test_search_papers_modes(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    # Hybrid mode
    res_hyb, col_hyb, label_hyb = search_papers(
        query="transformer",
        papers=papers,
        mode="hybrid",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=3,
    )
    assert col_hyb == "rrf_score"
    assert label_hyb == "RRF Ranking Score"
    assert "rrf_score" in res_hyb.columns
    assert len(res_hyb) <= 3

    # Semantic mode
    res_sem, col_sem, label_sem = search_papers(
        query="transformer",
        papers=papers,
        mode="semantic",
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=3,
    )
    assert col_sem == "similarity"
    assert label_sem == "Semantic Cosine Similarity"
    assert "similarity" in res_sem.columns

    # Lexical mode
    res_lex, col_lex, label_lex = search_papers(
        query="transformer",
        papers=papers,
        mode="lexical",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=3,
    )
    assert col_lex == "similarity"
    assert label_lex == "Lexical Similarity (TF-IDF)"
    assert "similarity" in res_lex.columns

    # UI mode string resolution
    res_ui, col_ui, label_ui = search_papers(
        query="transformer",
        papers=papers,
        mode="Hybrid (TF-IDF + Dense RRF)",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=3,
    )
    assert col_ui == "rrf_score"


def test_search_papers_fallback_when_semantic_missing(recommender):
    papers, vectorizer, tfidf_matrix = recommender

    # Hybrid requested without embeddings -> fallback to lexical
    res, score_col, label = search_papers(
        query="transformer",
        papers=papers,
        mode="hybrid",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=None,
    )
    assert score_col == "similarity"
    assert "Fallback" in label
    assert not res.empty

    # Semantic requested without embeddings -> fallback to lexical
    res2, score_col2, label2 = search_papers(
        query="transformer",
        papers=papers,
        mode="semantic",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=None,
    )
    assert score_col2 == "similarity"
    assert "Fallback" in label2
    assert not res2.empty


def test_search_papers_fallback_missing_vectorizer_raises(recommender):
    papers, _, _ = recommender
    with pytest.raises(ValueError, match="vectorizer and tfidf_matrix must be provided"):
        search_papers(
            query="transformer",
            papers=papers,
            mode="hybrid",
            semantic_embeddings=None,
            vectorizer=None,
            tfidf_matrix=None,
        )


def test_search_papers_invalid_mode_raises(recommender):
    papers, vectorizer, tfidf_matrix = recommender
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        search_papers("transformer", papers, mode="random_search")

    with pytest.raises(ValueError, match="mode must be a string"):
        search_papers("transformer", papers, mode=123)


def test_search_papers_empty_query(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    # Hybrid empty query
    res, col, _ = search_papers(
        query="   ",
        papers=papers,
        mode="hybrid",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
    )
    assert res.empty
    assert col == "rrf_score"

    # Semantic empty query
    res_s, col_s, _ = search_papers(
        query="",
        papers=papers,
        mode="semantic",
        semantic_embeddings=embeddings,
    )
    assert res_s.empty
    assert col_s == "similarity"


def test_search_papers_invalid_top_k(recommender):
    papers, vectorizer, tfidf_matrix = recommender
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search_papers("query", papers, top_k=0, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)


def test_search_papers_candidate_pool_size(recommender, mock_embedding_env):
    papers, vectorizer, tfidf_matrix = recommender
    _, embeddings, mock_model = mock_embedding_env

    res, col, _ = search_papers(
        query="transformer",
        papers=papers,
        mode="hybrid",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=embeddings,
        semantic_model=mock_model,
        top_k=2,
        candidate_pool_size=4,
    )
    assert not res.empty
    assert len(res) <= 2
    assert col == "rrf_score"


# ---------------------------------------------------------------------------
# Related Paper Recommendations (Item-to-Item)
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


# ---------------------------------------------------------------------------
# BibTeX Export Tests
# ---------------------------------------------------------------------------

def test_escape_bibtex_characters():
    assert escape_bibtex("Deep & Fast: 50% Speedup ($100) {v1}") == r"Deep \& Fast: 50\% Speedup (\$100) \{v1\}"
    assert escape_bibtex(None) == ""
    assert escape_bibtex(np.nan) == ""


def test_export_to_bibtex_empty():
    assert export_to_bibtex(pd.DataFrame()) == ""
    assert export_to_bibtex(None) == ""


def test_export_to_bibtex_structure(recommender):
    papers, _, _ = recommender
    sample = papers.head(2)
    bibtex = export_to_bibtex(sample)

    assert "@article{" in bibtex
    assert "title = {" in bibtex
    assert "year = {" in bibtex
    assert "journal = {" in bibtex
    assert "url = {" in bibtex
    assert "Accessed via ResearchMatch ML" in bibtex
    # Verify no fake author is generated
    assert "author = {" not in bibtex


def test_export_to_bibtex_with_author_if_present():
    df = pd.DataFrame([{
        "id": 101,
        "title": "Quantum & Neural Computation",
        "abstract": "A study on quantum neural nets.",
        "category": "Quantum ML",
        "year": 2024,
        "url": "https://example.com/quantum",
        "author": "Alice Smith & Bob Jones",
    }])

    bibtex = export_to_bibtex(df)
    assert "@article{quantum2024_101," in bibtex
    assert r"title = {Quantum \& Neural Computation}" in bibtex
    assert r"author = {Alice Smith \& Bob Jones}" in bibtex
    assert "year = {2024}" in bibtex


# ---------------------------------------------------------------------------
# Persistent Embedding Cache Tests
# ---------------------------------------------------------------------------

def test_compute_corpus_hash_deterministic(recommender):
    papers, _, _ = recommender
    h1 = compute_corpus_hash(papers, model_name="all-MiniLM-L6-v2")
    h2 = compute_corpus_hash(papers, model_name="all-MiniLM-L6-v2")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string


def test_compute_corpus_hash_different_content(recommender):
    papers, _, _ = recommender
    modified_papers = papers.copy()
    modified_papers.loc[0, "title"] = "A Completely Modified Paper Title"
    h_orig = compute_corpus_hash(papers)
    h_mod = compute_corpus_hash(modified_papers)
    assert h_orig != h_mod


def test_compute_corpus_hash_row_order_sensitive(recommender):
    papers, _, _ = recommender
    reversed_papers = papers.iloc[::-1].reset_index(drop=True)
    h_orig = compute_corpus_hash(papers)
    h_rev = compute_corpus_hash(reversed_papers)
    assert h_orig != h_rev


def test_compute_corpus_hash_different_models(recommender):
    papers, _, _ = recommender
    h_model1 = compute_corpus_hash(papers, model_name="all-MiniLM-L6-v2")
    h_model2 = compute_corpus_hash(papers, model_name="paraphrase-MiniLM-L3-v2")
    assert h_model1 != h_model2


def test_compute_corpus_hash_empty_dataset():
    empty_df = pd.DataFrame()
    h_empty = compute_corpus_hash(empty_df)
    assert isinstance(h_empty, str)
    assert len(h_empty) == 64


def test_save_and_validate_embeddings_roundtrip(tmp_path):
    target_file = tmp_path / "test_embeddings.npy"
    rng = np.random.default_rng(42)
    fake_embeddings = rng.standard_normal((10, 384)).astype(np.float32)

    saved_path = save_embeddings(fake_embeddings, target_file)
    assert saved_path == target_file
    assert target_file.is_file()

    with open(target_file, "rb") as f:
        loaded = np.load(f, allow_pickle=False)

    validated = validate_embeddings(loaded, expected_rows=10)
    assert np.allclose(fake_embeddings, validated)
    assert validated.dtype == np.float32


def test_save_embeddings_invalid_shapes_or_values(tmp_path):
    target_file = tmp_path / "bad_embeddings.npy"

    # 1D array should fail
    with pytest.raises(ValueError, match="must be a 2D array"):
        save_embeddings(np.ones(10, dtype=np.float32), target_file)

    # 3D array should fail
    with pytest.raises(ValueError, match="must be a 2D array"):
        save_embeddings(np.ones((2, 5, 10), dtype=np.float32), target_file)

    # NaN values should fail
    bad_arr = np.ones((5, 10), dtype=np.float32)
    bad_arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="contain NaN or infinite"):
        save_embeddings(bad_arr, target_file)

    # Infinite values should fail
    bad_arr[0, 0] = np.inf
    with pytest.raises(ValueError, match="contain NaN or infinite"):
        save_embeddings(bad_arr, target_file)


def test_validate_embeddings_errors():
    valid = np.ones((5, 10), dtype=np.float32)

    # Row count mismatch
    with pytest.raises(ValueError, match="row count .* does not match"):
        validate_embeddings(valid, expected_rows=6)

    # Zero feature dimension
    with pytest.raises(ValueError, match="feature dimension cannot be 0"):
        validate_embeddings(np.ones((5, 0), dtype=np.float32))

    # Integer dtype
    with pytest.raises(ValueError, match="Expected floating point dtype"):
        validate_embeddings(np.ones((5, 10), dtype=np.int32))

    # Non-ndarray
    with pytest.raises(ValueError, match="Loaded object is not a NumPy ndarray"):
        validate_embeddings([[1.0, 2.0], [3.0, 4.0]])


def test_load_or_compute_embeddings_cache_miss_saves(recommender, tmp_path):
    papers, _, _ = recommender
    mock_model = MagicMock()
    rng = np.random.default_rng(123)
    fake_vecs = rng.standard_normal((len(papers), 384)).astype(np.float32)
    mock_model.encode.return_value = fake_vecs

    embeddings = load_or_compute_embeddings(
        papers,
        model=mock_model,
        model_name="mock-model",
        cache_dir=tmp_path,
    )

    assert mock_model.encode.call_count == 1
    assert embeddings.shape == (len(papers), 384)
    assert np.allclose(embeddings, fake_vecs)

    # Verify file was written to tmp_path
    cache_files = list(tmp_path.glob("*.npy"))
    assert len(cache_files) == 1
    assert "mock-model" in cache_files[0].name


def test_load_or_compute_embeddings_cache_hit_avoids_recomputation(recommender, tmp_path):
    papers, _, _ = recommender
    mock_model = MagicMock()
    rng = np.random.default_rng(456)
    fake_vecs = rng.standard_normal((len(papers), 384)).astype(np.float32)
    mock_model.encode.return_value = fake_vecs

    # Run 1: Cache Miss
    emb1 = load_or_compute_embeddings(
        papers,
        model=mock_model,
        model_name="mock-hit-model",
        cache_dir=tmp_path,
    )
    assert mock_model.encode.call_count == 1

    # Run 2: Cache Hit (model.encode must NOT be called again)
    emb2 = load_or_compute_embeddings(
        papers,
        model=mock_model,
        model_name="mock-hit-model",
        cache_dir=tmp_path,
    )
    assert mock_model.encode.call_count == 1
    assert np.allclose(emb1, emb2)


def test_load_or_compute_embeddings_changed_dataset_invalidates_cache(recommender, tmp_path):
    papers, _, _ = recommender
    mock_model = MagicMock()
    rng = np.random.default_rng(789)
    fake_vecs1 = rng.standard_normal((len(papers), 384)).astype(np.float32)
    mock_model.encode.return_value = fake_vecs1

    # Run 1: Original papers
    load_or_compute_embeddings(
        papers,
        model=mock_model,
        model_name="mock-inval-model",
        cache_dir=tmp_path,
    )
    assert mock_model.encode.call_count == 1

    # Modify papers
    modified_papers = papers.copy()
    modified_papers.loc[0, "abstract"] = "A modified abstract that triggers invalidation."
    fake_vecs2 = rng.standard_normal((len(modified_papers), 384)).astype(np.float32)
    mock_model.encode.return_value = fake_vecs2

    # Run 2: Modified papers should cause cache miss and call model.encode again
    load_or_compute_embeddings(
        modified_papers,
        model=mock_model,
        model_name="mock-inval-model",
        cache_dir=tmp_path,
    )
    assert mock_model.encode.call_count == 2
    # Two distinct cache files should now exist
    cache_files = list(tmp_path.glob("*.npy"))
    assert len(cache_files) == 2


def test_load_or_compute_embeddings_corrupted_cache_recovery(recommender, tmp_path):
    papers, _, _ = recommender
    corpus_hash = compute_corpus_hash(papers, model_name="mock-corrupt-model")
    corrupt_file = tmp_path / f"embeddings_mock-corrupt-model_{corpus_hash}.npy"

    # Write invalid / corrupt bytes to simulate a broken cache
    corrupt_file.write_bytes(b"corrupted binary data that is not a valid numpy file")

    mock_model = MagicMock()
    rng = np.random.default_rng(999)
    fake_vecs = rng.standard_normal((len(papers), 384)).astype(np.float32)
    mock_model.encode.return_value = fake_vecs

    recovered = load_or_compute_embeddings(
        papers,
        model=mock_model,
        model_name="mock-corrupt-model",
        cache_dir=tmp_path,
    )

    # Should catch the corruption, log warning, recompute, and overwrite with valid data
    assert mock_model.encode.call_count == 1
    assert np.allclose(recovered, fake_vecs)
    assert corrupt_file.is_file()

    # Verify corrupt_file is now valid numpy
    with open(corrupt_file, "rb") as f:
        re_loaded = np.load(f, allow_pickle=False)
    assert np.allclose(re_loaded, fake_vecs)


def test_load_or_compute_embeddings_empty_dataset(tmp_path):
    empty_df = pd.DataFrame()
    mock_model = MagicMock()

    res = load_or_compute_embeddings(
        empty_df,
        model=mock_model,
        model_name="mock-model",
        cache_dir=tmp_path,
    )

    assert res.shape == (0, 384)
    assert mock_model.encode.call_count == 0
    # No cache file should be created for empty dataset
    assert len(list(tmp_path.glob("*.npy"))) == 0
