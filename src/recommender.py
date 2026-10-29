"""ResearchMatch ML Recommendation Engine.

Provides content-based and semantic retrieval for research papers:
1. TF-IDF + Cosine Similarity (Lexical Baseline)
2. Dense Sentence Embeddings via sentence-transformers (Semantic Search)
3. Reciprocal Rank Fusion (RRF) Hybrid Search
4. Item-to-Item Related Paper Recommendations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "papers.csv"
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RRF_K = 60

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
    results = results.sort_values(
        by=["similarity", "title"],
        ascending=[False, True],
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
    results = results.sort_values(
        by=["similarity", "title"],
        ascending=[False, True],
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
) -> pd.DataFrame:
    """Combine lexical (TF-IDF) and semantic (Dense) search using Reciprocal Rank Fusion.

    Reciprocal Rank Fusion (RRF) formula for paper d:
        RRF_score(d) = sum_{m in models} ( weight_m / (rrf_k + rank_m(d)) )

    where rank_m(d) is the 1-based rank of paper d in retriever m's candidate list.
    RRF operates on ordinal ranking rather than raw scores, making it robust against
    incomparable score distributions.

    NOTE: The 'rrf_score' is a ranking signal, NOT a probability or accuracy metric.
    """
    if not isinstance(query, str) or not query.strip():
        return papers.iloc[0:0].assign(
            similarity=pd.Series(dtype=float),
            rrf_score=pd.Series(dtype=float),
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Retrieve all matches from both engines
    lexical_results = recommend_papers(
        query, papers, vectorizer, tfidf_matrix, top_k=len(papers)
    )
    semantic_results = recommend_semantic(
        query, papers, semantic_embeddings, model=semantic_model, top_k=len(papers)
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

        if l_rank is not None:
            score += lexical_weight / (rrf_k + l_rank)
        if s_rank is not None:
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

    merged = merged.sort_values(
        by=["rrf_score", "semantic_score", "title"],
        ascending=[False, False, True],
    )

    return merged.head(top_k).reset_index(drop=True)


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
    results = results.sort_values(
        by=["similarity", "title"],
        ascending=[False, True],
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

