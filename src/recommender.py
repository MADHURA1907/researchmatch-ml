
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "papers.csv"

REQUIRED_COLUMNS = {
    "id", "title", "abstract", "category", "year", "url"
}


def load_papers(data_path=DEFAULT_DATA_PATH):
    """Load and validate the research-paper dataset."""
    papers = pd.read_csv(data_path)

    missing = REQUIRED_COLUMNS - set(papers.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    papers = papers.dropna(subset=["title", "abstract"]).copy()
    papers["title"] = papers["title"].astype(str).str.strip()
    papers["abstract"] = papers["abstract"].astype(str).str.strip()
    papers = papers[
        (papers["title"] != "") & (papers["abstract"] != "")
    ]

    if papers.empty:
        raise ValueError("No valid research papers found.")

    papers["category"] = papers["category"].fillna("Uncategorized")
    papers["url"] = papers["url"].fillna("")
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce")

    papers = papers.reset_index(drop=True)
    return papers


def build_recommender(papers):
    """Create TF-IDF vectors from paper titles and abstracts."""
    
    documents = (
        papers["title"].fillna("").astype(str)
        + " "
        + papers["abstract"].fillna("").astype(str)
        + " "
        + papers["category"].fillna("").astype(str)
    ).tolist()


    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )

    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def recommend_papers(query, papers, vectorizer, matrix, top_k=5):
    """Rank papers by cosine similarity to a user's query."""
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
