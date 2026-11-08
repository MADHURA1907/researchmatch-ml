"""ResearchMatch ML Recommendation Engine.

Provides content-based and semantic retrieval for research papers:
1. TF-IDF + Cosine Similarity (Lexical Baseline)
2. Dense Sentence Embeddings via sentence-transformers (Semantic Search)
3. Reciprocal Rank Fusion (RRF) Hybrid Search
4. Item-to-Item Related Paper Recommendations
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "papers.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "embeddings_cache"
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_POOL_SIZE = 50

REQUIRED_COLUMNS = {"id", "title", "abstract", "category", "year", "url"}

# Global model cache to avoid re-loading weights in the same process
_MODEL_CACHE: Dict[str, Any] = {}


def normalize_text(text: Any) -> str:
    """Normalize input text by trimming and collapsing multiple spaces."""
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).strip().split())


def load_papers(data_path: Union[str, Path] = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load, validate, normalize, and deduplicate research papers from CSV."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    papers = pd.read_csv(data_path)

    missing = REQUIRED_COLUMNS - set(papers.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Clean title and abstract
    papers["title"] = papers["title"].apply(normalize_text)
    papers["abstract"] = papers["abstract"].apply(normalize_text)
    papers = papers[(papers["title"] != "") & (papers["abstract"] != "")].copy()

    if papers.empty:
        raise ValueError("No valid research papers found.")

    # Deduplicate by normalized title and paper ID
    papers["_norm_title"] = (
        papers["title"]
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .apply(normalize_text)
    )
    papers = papers.drop_duplicates(subset=["_norm_title"]).drop_duplicates(subset=["id"])
    papers = papers.drop(columns=["_norm_title"])

    # Fill and format metadata
    papers["category"] = papers["category"].fillna("Uncategorized").astype(str).str.strip()
    papers["category"] = papers["category"].replace("", "Uncategorized")
    papers["url"] = papers["url"].fillna("").astype(str).str.strip()
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce")

    return papers.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lexical (TF-IDF) Baseline
# ---------------------------------------------------------------------------

def _build_paper_documents(papers: pd.DataFrame) -> List[str]:
    """Combine title, abstract, and category into a single document string."""
    return (
        papers["title"].fillna("").astype(str)
        + " "
        + papers["abstract"].fillna("").astype(str)
        + " "
        + papers["category"].fillna("").astype(str)
    ).tolist()


def build_recommender(papers: pd.DataFrame) -> Tuple[TfidfVectorizer, Any]:
    """Create TF-IDF vectorizer and document-term matrix from paper metadata."""
    documents = _build_paper_documents(papers)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def recommend_papers(
    query: str,
    papers: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank papers using TF-IDF and cosine similarity (Lexical Baseline).

    Similarity scores represent cosine angle between query and document vectors,
    bounded between 0.0 and 1.0. They indicate vocabulary overlap, not paper quality.
    """
    if not isinstance(query, str) or not query.strip():
        return papers.iloc[0:0].assign(similarity=pd.Series(dtype=float))

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vector, matrix).ravel()

    results = papers.copy()
    results["similarity"] = scores

    results = results[results["similarity"] > 0]
    sort_cols = ["similarity", "title"]
    sort_asc = [False, True]
    if "id" in results.columns:
        sort_cols.append("id")
        sort_asc.append(True)
    results = results.sort_values(
        by=sort_cols,
        ascending=sort_asc,
    )

    return results.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Semantic Search (Dense Embeddings)
# ---------------------------------------------------------------------------

def get_semantic_model(model_name: str = DEFAULT_SEMANTIC_MODEL) -> Any:
    """Load and cache a sentence-transformers embedding model.

    Handles offline environments or missing packages gracefully.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' library is required for semantic search. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    try:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load semantic model '{model_name}': {exc}. "
            "Check network connectivity or local model cache."
        ) from exc


def build_semantic_embeddings(
    papers: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
) -> np.ndarray:
    """Compute normalized dense embeddings for paper titles, abstracts, and categories."""
    if model is None:
        model = get_semantic_model(model_name)

    documents = [
        f"Title: {title}. Abstract: {abstract} Category: {category}"
        for title, abstract, category in zip(
            papers["title"].fillna(""),
            papers["abstract"].fillna(""),
            papers["category"].fillna(""),
        )
    ]

    embeddings = model.encode(
        documents,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Persistent Semantic Embedding Cache
# ---------------------------------------------------------------------------

def compute_corpus_hash(
    papers: pd.DataFrame,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for paper content and model identity.

    Accounts for title, abstract, category, row ordering, and the embedding model identifier.
    """
    hasher = hashlib.sha256()
    hasher.update(str(model_name).strip().encode("utf-8"))
    hasher.update(b"\0")

    if papers is None or papers.empty:
        hasher.update(b"__empty__")
        return hasher.hexdigest()

    for _, row in papers.iterrows():
        title = normalize_text(row.get("title", ""))
        abstract = normalize_text(row.get("abstract", ""))
        category = normalize_text(row.get("category", ""))
        doc = f"Title: {title}. Abstract: {abstract} Category: {category}"
        hasher.update(doc.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def save_embeddings(
    embeddings: np.ndarray,
    cache_path: Union[str, Path],
) -> Path:
    """Save embeddings to disk safely using atomic file replacement.

    Validates array shape, dtype, and finiteness before saving.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Embeddings contain NaN or infinite values.")

    # Write to a temporary file in the same directory, then atomically replace
    temp_file = cache_path.with_suffix(f".tmp.{os.getpid()}_{np.random.randint(1e9)}.npy")
    try:
        with open(temp_file, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        temp_file.replace(cache_path)
    except Exception:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise

    return cache_path


def validate_embeddings(
    embeddings: Any,
    expected_rows: Optional[int] = None,
) -> np.ndarray:
    """Validate loaded embeddings array for type, shape, and finiteness."""
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Loaded object is not a NumPy ndarray.")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(f"Expected floating point dtype, got {embeddings.dtype}.")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}.")
    if expected_rows is not None and embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"expected count ({expected_rows})."
        )
    if embeddings.shape[1] == 0:
        raise ValueError("Embedding feature dimension cannot be 0.")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Loaded embeddings contain NaN or infinite values.")
    return embeddings.astype(np.float32, copy=False)


def load_or_compute_embeddings(
    papers: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force_compute: bool = False,
) -> np.ndarray:
    """Load cached embeddings if valid; otherwise compute, validate, and persist them.

    Returns float32 embeddings array of shape (len(papers), embedding_dim).
    """
    if papers is None or papers.empty:
        return np.empty((0, 384), dtype=np.float32)

    cache_dir = Path(cache_dir)
    corpus_hash = compute_corpus_hash(papers, model_name=model_name)
    safe_model = re.sub(r"[^\w\-]", "_", str(model_name).strip())
    cache_file = cache_dir / f"embeddings_{safe_model}_{corpus_hash}.npy"

    if not force_compute and cache_file.is_file():
        try:
            with open(cache_file, "rb") as f:
                loaded = np.load(f, allow_pickle=False)
            validated = validate_embeddings(loaded, expected_rows=len(papers))
            logger.info("Loaded valid semantic embeddings from cache: %s", cache_file)
            return validated
        except Exception as exc:
            logger.warning(
                "Invalid or corrupted embedding cache at %s (%s). Recomputing...",
                cache_file,
                exc,
            )
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass

    # Cache miss or invalid cache: compute embeddings
    embeddings = build_semantic_embeddings(papers, model=model, model_name=model_name)
    validated = validate_embeddings(embeddings, expected_rows=len(papers))

    try:
        save_embeddings(validated, cache_file)
        logger.info("Persisted semantic embeddings to cache: %s", cache_file)
    except Exception as exc:
        logger.warning("Failed to persist embeddings to cache at %s: %s", cache_file, exc)

    return validated


def recommend_semantic(
    query: str,
    papers: pd.DataFrame,
    embeddings: np.ndarray,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank papers using dense semantic embeddings and cosine similarity.

    Similarity scores represent dot-product cosine similarity between normalized
    query and document embeddings in the embedding space.
    """
    if not isinstance(query, str) or not query.strip():
        return papers.iloc[0:0].assign(similarity=pd.Series(dtype=float))

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if len(papers) != len(embeddings):
        raise ValueError(
            f"Mismatch between papers ({len(papers)}) and embeddings ({len(embeddings)})."
        )

    if model is None:
        model = get_semantic_model(model_name)

    query_embedding = model.encode(
        [query.strip()],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    scores = np.dot(embeddings, query_embedding[0])

    results = papers.copy()
    results["similarity"] = np.clip(scores, 0.0, 1.0)

    # Filter out near-zero or negative similarity
    results = results[results["similarity"] > 0]
    sort_cols = ["similarity", "title"]
    sort_asc = [False, True]
    if "id" in results.columns:
        sort_cols.append("id")
        sort_asc.append(True)
    results = results.sort_values(
        by=sort_cols,
        ascending=sort_asc,
    )

    return results.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
# ---------------------------------------------------------------------------

def recommend_hybrid(
    query: str,
    papers: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: Any,
    semantic_embeddings: np.ndarray,
    semantic_model: Optional[Any] = None,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    candidate_pool_size: Optional[int] = None,
) -> pd.DataFrame:
    """Combine lexical (TF-IDF) and semantic (Dense) search using Reciprocal Rank Fusion.

    Reciprocal Rank Fusion (RRF) formula for paper d:
        RRF_score(d) = sum_{m in models} ( weight_m / (rrf_k + rank_m(d)) )

    where rank_m(d) is the 1-based rank of paper d in retriever m's candidate list.
    RRF operates on ordinal ranking rather than raw scores, making it robust against
    incomparable score distributions.

    Candidate pooling: When `candidate_pool_size` is specified, only the top candidates
    from each retriever are merged, improving retrieval speed on large corpora while
    ensuring at least `top_k` candidates are evaluated.

    NOTE: The 'rrf_score' is a ranking signal, NOT a probability or accuracy metric.
    """
    if not isinstance(query, str) or not query.strip():
        return papers.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            rrf_score=pd.Series(dtype=float),
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    if lexical_weight < 0 or semantic_weight < 0:
        raise ValueError("lexical_weight and semantic_weight must be non-negative.")

    if lexical_weight == 0 and semantic_weight == 0:
        raise ValueError("At least one of lexical_weight or semantic_weight must be positive.")

    if candidate_pool_size is not None and candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be at least 1.")

    if papers.empty:
        return papers.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            rrf_score=pd.Series(dtype=float),
        )

    # Determine candidate pool depth
    if candidate_pool_size is None or candidate_pool_size >= len(papers):
        pool_depth = len(papers)
    else:
        pool_depth = min(len(papers), max(top_k, candidate_pool_size))

    # Retrieve candidates from both engines
    lexical_results = recommend_papers(
        query, papers, vectorizer, tfidf_matrix, top_k=pool_depth
    )
    semantic_results = recommend_semantic(
        query, papers, semantic_embeddings, model=semantic_model, top_k=pool_depth
    )

    if lexical_results.empty and semantic_results.empty:
        return papers.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            rrf_score=pd.Series(dtype=float),
        )

    # Build lookup dictionaries for rank and raw scores
    lexical_ranks: Dict[Any, int] = {}
    lexical_scores: Dict[Any, float] = {}
    for rank_idx, (_, row) in enumerate(lexical_results.iterrows(), start=1):
        pid = row["id"]
        lexical_ranks[pid] = rank_idx
        lexical_scores[pid] = float(row["similarity"])

    semantic_ranks: Dict[Any, int] = {}
    semantic_scores: Dict[Any, float] = {}
    for rank_idx, (_, row) in enumerate(semantic_results.iterrows(), start=1):
        pid = row["id"]
        semantic_ranks[pid] = rank_idx
        semantic_scores[pid] = float(row["similarity"])

    candidate_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())

    records = []
    for pid in candidate_ids:
        score = 0.0
        l_rank = lexical_ranks.get(pid)
        s_rank = semantic_ranks.get(pid)

        if l_rank is not None and lexical_weight > 0:
            score += lexical_weight / (rrf_k + l_rank)
        if s_rank is not None and semantic_weight > 0:
            score += semantic_weight / (rrf_k + s_rank)

        records.append({
            "id": pid,
            "rrf_score": score,
            "lexical_rank": l_rank,
            "semantic_rank": s_rank,
            "lexical_score": lexical_scores.get(pid, 0.0),
            "semantic_score": semantic_scores.get(pid, 0.0),
        })

    score_df = pd.DataFrame(records)
    merged = pd.merge(papers, score_df, on="id", how="inner")

    # Set similarity to rrf_score for interface compatibility
    merged["similarity"] = merged["rrf_score"]

    sort_cols = ["rrf_score", "semantic_score", "title"]
    sort_asc = [False, False, True]
    if "id" in merged.columns:
        sort_cols.append("id")
        sort_asc.append(True)

    merged = merged.sort_values(
        by=sort_cols,
        ascending=sort_asc,
    )

    return merged.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified Search Dispatcher
# ---------------------------------------------------------------------------

def search_papers(
    query: str,
    papers: pd.DataFrame,
    mode: str = "hybrid",
    top_k: int = 5,
    vectorizer: Optional[TfidfVectorizer] = None,
    tfidf_matrix: Optional[Any] = None,
    semantic_embeddings: Optional[np.ndarray] = None,
    semantic_model: Optional[Any] = None,
    rrf_k: int = DEFAULT_RRF_K,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    candidate_pool_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """Unified retrieval dispatcher for Lexical, Semantic, and Hybrid search.

    Routes queries to the appropriate search engine based on `mode`:
    - 'hybrid': Reciprocal Rank Fusion of TF-IDF and Semantic embeddings.
    - 'semantic': Dense embedding cosine similarity via sentence-transformers.
    - 'lexical' (or 'tfidf'): Sparse TF-IDF n-gram cosine similarity.

    Gracefully falls back to lexical retrieval if semantic embeddings are unavailable
    when 'hybrid' or 'semantic' mode is requested.

    Returns:
        Tuple of (results_df, score_column_name, metric_label):
        - results_df: pd.DataFrame containing the ranked paper metadata and scores.
        - score_column_name: Name of the primary ranking score column ("rrf_score" or "similarity").
        - metric_label: Human-readable label for UI rendering (e.g. "RRF Ranking Score").
    """
    if not isinstance(mode, str):
        raise ValueError(f"mode must be a string, got {type(mode).__name__}.")

    clean_mode = mode.strip().lower()
    if "hybrid" in clean_mode:
        resolved_mode = "hybrid"
    elif "semantic" in clean_mode:
        resolved_mode = "semantic"
    elif "lexical" in clean_mode or "tfidf" in clean_mode or "tf-idf" in clean_mode:
        resolved_mode = "lexical"
    else:
        raise ValueError(
            f"Invalid retrieval mode '{mode}'. Supported modes are 'hybrid', 'semantic', 'lexical'."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Handle empty query early
    if not isinstance(query, str) or not query.strip():
        if resolved_mode == "hybrid":
            if semantic_embeddings is None:
                logger.warning(
                    "Semantic embeddings unavailable. Falling back to Lexical (TF-IDF) search."
                )
                return (
                    papers.iloc[0:0].assign(similarity=pd.Series(dtype=float)),
                    "similarity",
                    "Lexical Similarity (TF-IDF - Fallback)",
                )
            return (
                papers.iloc[0:0].assign(
                    similarity=pd.Series(dtype=float),
                    rrf_score=pd.Series(dtype=float),
                ),
                "rrf_score",
                "RRF Ranking Score",
            )
        elif resolved_mode == "semantic":
            if semantic_embeddings is None:
                logger.warning(
                    "Semantic embeddings unavailable. Falling back to Lexical (TF-IDF) search."
                )
                return (
                    papers.iloc[0:0].assign(similarity=pd.Series(dtype=float)),
                    "similarity",
                    "Lexical Similarity (TF-IDF - Fallback)",
                )
            return (
                papers.iloc[0:0].assign(similarity=pd.Series(dtype=float)),
                "similarity",
                "Semantic Cosine Similarity",
            )
        else:
            return (
                papers.iloc[0:0].assign(similarity=pd.Series(dtype=float)),
                "similarity",
                "Lexical Similarity (TF-IDF)",
            )

    # Route according to mode with graceful fallback
    if resolved_mode == "hybrid":
        if semantic_embeddings is None:
            logger.warning(
                "Semantic embeddings unavailable for hybrid retrieval. "
                "Falling back to Lexical (TF-IDF) search."
            )
            if vectorizer is None or tfidf_matrix is None:
                raise ValueError(
                    "vectorizer and tfidf_matrix must be provided when falling back to lexical search."
                )
            results = recommend_papers(
                query=query,
                papers=papers,
                vectorizer=vectorizer,
                matrix=tfidf_matrix,
                top_k=top_k,
            )
            return results, "similarity", "Lexical Similarity (TF-IDF - Fallback)"

        if vectorizer is None or tfidf_matrix is None:
            raise ValueError(
                "vectorizer and tfidf_matrix must be provided for hybrid retrieval."
            )

        results = recommend_hybrid(
            query=query,
            papers=papers,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=semantic_embeddings,
            semantic_model=semantic_model,
            top_k=top_k,
            rrf_k=rrf_k,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            candidate_pool_size=candidate_pool_size,
        )
        return results, "rrf_score", "RRF Ranking Score"

    elif resolved_mode == "semantic":
        if semantic_embeddings is None:
            logger.warning(
                "Semantic embeddings unavailable for semantic retrieval. "
                "Falling back to Lexical (TF-IDF) search."
            )
            if vectorizer is None or tfidf_matrix is None:
                raise ValueError(
                    "vectorizer and tfidf_matrix must be provided when falling back to lexical search."
                )
            results = recommend_papers(
                query=query,
                papers=papers,
                vectorizer=vectorizer,
                matrix=tfidf_matrix,
                top_k=top_k,
            )
            return results, "similarity", "Lexical Similarity (TF-IDF - Fallback)"

        results = recommend_semantic(
            query=query,
            papers=papers,
            embeddings=semantic_embeddings,
            model=semantic_model,
            top_k=top_k,
        )
        return results, "similarity", "Semantic Cosine Similarity"

    else:  # resolved_mode == "lexical"
        if vectorizer is None or tfidf_matrix is None:
            raise ValueError(
                "vectorizer and tfidf_matrix must be provided for lexical retrieval."
            )
        results = recommend_papers(
            query=query,
            papers=papers,
            vectorizer=vectorizer,
            matrix=tfidf_matrix,
            top_k=top_k,
        )
        return results, "similarity", "Lexical Similarity (TF-IDF)"


# ---------------------------------------------------------------------------
# Related Paper Recommendations (Item-to-Item)
# ---------------------------------------------------------------------------

def get_related_papers(
    paper_id: Any,
    papers: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    matrix: Optional[Any] = None,
    top_k: int = 3,
    method: str = "semantic",
) -> pd.DataFrame:
    """Find the most similar papers to a given target paper (Item-to-Item).

    Excludes the target paper itself from recommendations.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Match paper by ID (handling int/string differences)
    match_indices = papers.index[papers["id"].astype(str) == str(paper_id)].tolist()
    if not match_indices:
        raise ValueError(f"Paper with ID '{paper_id}' not found in dataset.")

    target_idx = match_indices[0]

    if method == "semantic" and embeddings is not None:
        target_vec = embeddings[target_idx]
        scores = np.dot(embeddings, target_vec)
    elif vectorizer is not None and matrix is not None:
        target_vec = matrix[target_idx]
        scores = cosine_similarity(target_vec, matrix).ravel()
    elif embeddings is not None:
        target_vec = embeddings[target_idx]
        scores = np.dot(embeddings, target_vec)
    else:
        raise ValueError("Either embeddings or (vectorizer, matrix) must be provided.")

    results = papers.copy()
    results["similarity"] = np.clip(scores, 0.0, 1.0)

    # Exclude the target paper itself
    results = results[results.index != target_idx]
    results = results[results["similarity"] > 0]
    sort_cols = ["similarity", "title"]
    sort_asc = [False, True]
    if "id" in results.columns:
        sort_cols.append("id")
        sort_asc.append(True)
    results = results.sort_values(
        by=sort_cols,
        ascending=sort_asc,
    )

    return results.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# BibTeX Export Utility
# ---------------------------------------------------------------------------

def escape_bibtex(text: Any) -> str:
    """Escape special BibTeX characters in text values."""
    if text is None or pd.isna(text):
        return ""
    val = str(text)
    val = val.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, rep in replacements.items():
        val = val.replace(char, rep)
    return val


def export_to_bibtex(papers: pd.DataFrame) -> str:
    """Generate valid BibTeX citation entries for research papers.

    Only uses metadata actually available in the dataset (does NOT invent missing authors).
    """
    if papers is None or papers.empty:
        return ""

    entries = []
    for _, paper in papers.iterrows():
        pid = paper.get("id", "unknown")
        title = paper.get("title", "Untitled")
        words = str(title).split()
        first_word = "".join(filter(str.isalnum, words[0])) if words else "paper"
        year_val = paper.get("year")
        try:
            year_str = str(int(year_val)) if pd.notna(year_val) else ""
        except (ValueError, TypeError):
            year_str = ""
        cite_key = f"{first_word.lower()}{year_str}_{pid}"

        fields = [f"  title = {{{escape_bibtex(title)}}}"]

        if "author" in paper and pd.notna(paper["author"]) and str(paper["author"]).strip():
            fields.append(f"  author = {{{escape_bibtex(paper['author'])}}}")
        elif "authors" in paper and pd.notna(paper["authors"]) and str(paper["authors"]).strip():
            fields.append(f"  author = {{{escape_bibtex(paper['authors'])}}}")

        if year_str:
            fields.append(f"  year = {{{year_str}}}")

        category = paper.get("category")
        if pd.notna(category) and str(category).strip():
            fields.append(f"  journal = {{{escape_bibtex(category)}}}")

        url = paper.get("url")
        if pd.notna(url) and str(url).strip():
            fields.append(f"  url = {{{str(url).strip()}}}")

        fields.append("  note = {Accessed via ResearchMatch ML}")

        field_lines = ",\n".join(fields)
        entry = f"@article{{{cite_key},\n{field_lines}\n}}"
        entries.append(entry)

    return "\n\n".join(entries)
