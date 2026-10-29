"""ResearchMatch ML - Academic Paper Discovery Platform.

An interactive research paper recommendation and exploration system
supporting lexical (TF-IDF), semantic (all-MiniLM-L6-v2), and hybrid (RRF) retrieval.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.recommender import (
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_MODEL,
    build_recommender,
    build_semantic_embeddings,
    export_to_bibtex,
    get_related_papers,
    get_semantic_model,
    load_papers,
    recommend_hybrid,
    recommend_papers,
    recommend_semantic,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMatch ML | Paper Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished typography, badges, and cards
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-category {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #EEF2F6;
        color: #1E3A8A;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-year {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 9999px;
    }
    .score-metric {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F766E;
    }
    .disclaimer-text {
        font-size: 0.8rem;
        color: #94A3B8;
        font-style: italic;
    }
    div.stButton > button {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "reading_list" not in st.session_state:
    st.session_state.reading_list = set()

if "comparison_list" not in st.session_state:
    st.session_state.comparison_list = []

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "related_paper_id" not in st.session_state:
    st.session_state.related_paper_id = None

# ---------------------------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------------------------

@st.cache_data
def get_papers() -> pd.DataFrame:
    """Load and validate the curated research paper dataset."""
    return load_papers()

@st.cache_resource
def get_tfidf_model(papers: pd.DataFrame):
    """Fit and cache the TF-IDF lexical retrieval baseline."""
    return build_recommender(papers)

@st.cache_resource
def get_semantic_engine(papers: pd.DataFrame):
    """Load sentence embedding model and compute paper embeddings.

    Returns (model, embeddings, error_message).
    """
    try:
        model = get_semantic_model(DEFAULT_SEMANTIC_MODEL)
        embeddings = build_semantic_embeddings(papers, model=model)
        return model, embeddings, None
    except Exception as exc:
        logger.warning("Semantic search engine unavailable: %s", exc)
        return None, None, str(exc)

# Load data safely
try:
    papers = get_papers()
    vectorizer, tfidf_matrix = get_tfidf_model(papers)
except Exception as exc:
    st.error(f"Failed to load research dataset: {exc}")
    st.stop()

# Initialize semantic model
semantic_model, semantic_embeddings, semantic_error = get_semantic_engine(papers)
semantic_available = (semantic_model is not None and semantic_embeddings is not None)

# Derive dataset ranges safely
valid_years = papers["year"].dropna().astype(int)
min_year = int(valid_years.min()) if not valid_years.empty else 2000
max_year = int(valid_years.max()) if not valid_years.empty else 2026
all_categories = sorted(papers["category"].dropna().unique().tolist())

# ---------------------------------------------------------------------------
# Sidebar Navigation & Search Parameters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### ⚙️ Search Controls")

# Retrieval Mode Selection
search_modes = ["Hybrid (TF-IDF + Dense RRF)", "Semantic (all-MiniLM-L6-v2)", "Lexical Baseline (TF-IDF)"]
if not semantic_available:
    st.sidebar.warning(
        f"⚠️ Semantic search unavailable ({semantic_error}). Falling back to Lexical TF-IDF baseline."
    )
    mode_index = 2
else:
    mode_index = 0

selected_mode = st.sidebar.selectbox(
    "Retrieval Engine",
    options=search_modes,
    index=mode_index,
    help="Select the information retrieval model used to score and rank papers.",
    disabled=not semantic_available,
)

if "Hybrid" in selected_mode:
    st.sidebar.caption(
        "Combines lexical keyword matching and dense sentence embeddings via Reciprocal Rank Fusion (RRF)."
    )
elif "Semantic" in selected_mode:
    st.sidebar.caption(
        "Calculates cosine similarity between 384-dimensional dense sentence embeddings."
    )
else:
    st.sidebar.caption(
        "Calculates cosine similarity on n-gram TF-IDF document-term vectors."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# Category Filter
selected_categories = st.sidebar.multiselect(
    "Filter by Category",
    options=all_categories,
    default=[],
    help="Leave empty to include all research areas.",
)

# Year Filter
if min_year < max_year:
    selected_year_range = st.sidebar.slider(
        "Publication Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
else:
    selected_year_range = (min_year, max_year)

# Sorting
sort_options = [
    "Relevance (Best match first)",
    "Publication Year (Newest first)",
    "Publication Year (Oldest first)",
    "Title (A to Z)",
]
selected_sort = st.sidebar.selectbox("Sort Results By", options=sort_options)

# Result Count Slider
top_k = st.sidebar.slider(
    "Max Results",
    min_value=1,
    max_value=len(papers),
    value=min(5, len(papers)),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.write(f"• **{len(papers)}** landmark papers indexed")
st.sidebar.write(f"• **{len(all_categories)}** research categories")
st.sidebar.write(f"• **{len(st.session_state.reading_list)}** saved to reading list")
st.sidebar.write(f"• **{len(st.session_state.comparison_list)}** selected for comparison")

# ---------------------------------------------------------------------------
# Main Layout: Tabs
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">📚 ResearchMatch ML</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Discover, compare, and explore seminal machine learning papers with hybrid retrieval.</div>',
    unsafe_allow_html=True,
)

tab_search, tab_reading_list, tab_compare = st.tabs([
    "🔍 Search & Discover",
    f"📑 Reading List ({len(st.session_state.reading_list)})",
    f"⚖️ Paper Comparison ({len(st.session_state.comparison_list)})",
])

# ---------------------------------------------------------------------------
# Tab 1: Search & Discover
# ---------------------------------------------------------------------------

with tab_search:
    # Example Query Chips
    st.markdown("**Try a sample research topic:**")
    chip_cols = st.columns(5)
    example_queries = [
        "Attention & Transformers",
        "Diffusion Models",
        "Biomedical Segmentation",
        "LoRA Adaptation",
        "Stochastic Optimization",
    ]

    for col, example in zip(chip_cols, example_queries):
        if col.button(example, use_container_width=True):
            st.session_state.search_query = example

    # Search Box
    search_query = st.text_input(
        "What would you like to research?",
        value=st.session_state.search_query,
        placeholder="e.g., deep residual learning for image recognition",
        help="Enter research keywords, problem descriptions, or techniques.",
        key="main_search_input",
    )
    # Sync state
    st.session_state.search_query = search_query

    # Execute Search
    if search_query.strip():
        # Retrieve candidate papers using selected engine
        if "Hybrid" in selected_mode and semantic_available:
            score_col_name = "rrf_score"
            metric_label = "RRF Ranking Score"
            results = recommend_hybrid(
                query=search_query,
                papers=papers,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                semantic_embeddings=semantic_embeddings,
                semantic_model=semantic_model,
                top_k=len(papers),
                rrf_k=DEFAULT_RRF_K,
            )
        elif "Semantic" in selected_mode and semantic_available:
            score_col_name = "similarity"
            metric_label = "Semantic Similarity"
            results = recommend_semantic(
                query=search_query,
                papers=papers,
                embeddings=semantic_embeddings,
                model=semantic_model,
                top_k=len(papers),
            )
        else:
            score_col_name = "similarity"
            metric_label = "Lexical Similarity (TF-IDF)"
            results = recommend_papers(
                query=search_query,
                papers=papers,
                vectorizer=vectorizer,
                matrix=tfidf_matrix,
                top_k=len(papers),
            )

        # Apply Category Filter
        if selected_categories:
            results = results[results["category"].isin(selected_categories)]

        # Apply Publication Year Filter
        results = results[
            (results["year"] >= selected_year_range[0])
            & (results["year"] <= selected_year_range[1])
        ]

        # Apply Sorting
        if selected_sort == "Publication Year (Newest first)":
            results = results.sort_values(by="year", ascending=False)
        elif selected_sort == "Publication Year (Oldest first)":
            results = results.sort_values(by="year", ascending=True)
        elif selected_sort == "Title (A to Z)":
            results = results.sort_values(by="title", ascending=True)
        # Default is already sorted by relevance

        # Slice to Top K
        displayed_results = results.head(top_k).reset_index(drop=True)

        st.markdown(f"### Matching Papers ({len(displayed_results)})")

        if displayed_results.empty:
            st.info(
                "No papers matched your criteria. Try broadening your query, "
                "or resetting the category and publication-year filters."
            )
        else:
            # Metrics bar
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieved Papers", len(displayed_results))
            m2.metric("Research Areas", displayed_results["category"].nunique())
            best_score = displayed_results[score_col_name].max()
            if "Hybrid" in selected_mode:
                m3.metric("Top RRF Score", f"{best_score:.4f}")
            else:
                m3.metric("Top Cosine Match", f"{best_score:.1%}")
            m4.metric("Active Model", selected_mode.split()[0])

            # Export Buttons
            export_cols = st.columns([1, 1, 4])
            csv_data = displayed_results.to_csv(index=False)
            bibtex_data = export_to_bibtex(displayed_results)

            export_cols[0].download_button(
                "📥 Export CSV",
                data=csv_data,
                file_name="researchmatch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            export_cols[1].download_button(
                "📥 Export BibTeX",
                data=bibtex_data,
                file_name="researchmatch_results.bib",
                mime="text/plain",
                use_container_width=True,
            )

            st.caption(
                "⚠️ Note: Scores denote relative algorithmic relevance to your search query. "
                "They do not reflect paper citation impact or peer-review quality."
            )

            # Paper Cards
            for idx, paper in displayed_results.iterrows():
                pid = paper["id"]
                with st.container(border=True):
                    card_title_col, card_badge_col = st.columns([3, 1])

                    with card_title_col:
                        st.markdown(f"#### {paper['title']}")
                        year_display = int(paper["year"]) if pd.notna(paper["year"]) else "Year unknown"
                        st.markdown(
                            f'<span class="badge-category">{paper["category"]}</span>'
                            f'<span class="badge-year">{year_display}</span>',
                            unsafe_allow_html=True,
                        )

                    with card_badge_col:
                        score_val = paper[score_col_name]
                        if "Hybrid" in selected_mode:
                            st.metric(metric_label, f"{score_val:.4f}")
                        else:
                            st.metric(metric_label, f"{score_val:.1%}")

                    # Abstract
                    with st.expander("📄 Full Abstract & Metadata", expanded=False):
                        st.write(paper["abstract"])
                        # Authors field truthfulness
                        if "author" in paper and pd.notna(paper["author"]):
                            st.caption(f"**Authors:** {paper['author']}")
                        else:
                            st.caption("**Authors:** Not provided in dataset metadata")

                        # RRF Component ranks if in hybrid mode
                        if "Hybrid" in selected_mode:
                            l_rank = paper.get("lexical_rank")
                            s_rank = paper.get("semantic_rank")
                            l_display = f"#{l_rank}" if pd.notna(l_rank) else "Not in top lexical"
                            s_display = f"#{s_rank}" if pd.notna(s_rank) else "Not in top semantic"
                            st.caption(
                                f"**Rank Breakdown:** Lexical Rank: {l_display} | Semantic Rank: {s_display}"
                            )

                    # Action Buttons Row
                    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.8, 1.8, 2.5])

                    # Direct link
                    if paper["url"]:
                        btn_col1.link_button("Source ↗", paper["url"], use_container_width=True)

                    # Reading List toggle
                    is_saved = pid in st.session_state.reading_list
                    save_label = "🔖 Remove" if is_saved else "📌 Save"
                    if btn_col2.button(save_label, key=f"save_{pid}_{idx}", use_container_width=True):
                        if is_saved:
                            st.session_state.reading_list.remove(pid)
                        else:
                            st.session_state.reading_list.add(pid)
                        st.rerun()

                    # Comparison toggle
                    is_compared = pid in st.session_state.comparison_list
                    cmp_label = "⚖️ Uncompare" if is_compared else "⚖️ Compare"
                    if btn_col3.button(cmp_label, key=f"cmp_{pid}_{idx}", use_container_width=True):
                        if is_compared:
                            st.session_state.comparison_list.remove(pid)
                        else:
                            if len(st.session_state.comparison_list) >= 3:
                                st.warning("Maximum of 3 papers can be compared simultaneously.")
                            else:
                                st.session_state.comparison_list.append(pid)
                        st.rerun()

                    # Related papers button
                    if btn_col4.button("🔗 Related Papers", key=f"rel_{pid}_{idx}", use_container_width=True):
                        st.session_state.related_paper_id = (
                            None if st.session_state.related_paper_id == pid else pid
                        )
                        st.rerun()

                    # Inline Related Papers
                    if st.session_state.related_paper_id == pid:
                        with st.container(border=True):
                            st.markdown(f"**Papers most similar to:** *{paper['title']}*")
                            try:
                                related = get_related_papers(
                                    paper_id=pid,
                                    papers=papers,
                                    embeddings=semantic_embeddings if semantic_available else None,
                                    vectorizer=vectorizer if not semantic_available else None,
                                    matrix=tfidf_matrix if not semantic_available else None,
                                    top_k=3,
                                    method="semantic" if semantic_available else "tfidf",
                                )
                                for _, r_paper in related.iterrows():
                                    r_col1, r_col2 = st.columns([3, 1])
                                    r_col1.write(f"• **{r_paper['title']}** ({r_paper['category']}, {int(r_paper['year'])})")
                                    r_col2.write(f"Similarity: **{r_paper['similarity']:.1%}**")
                            except Exception as rel_err:
                                st.error(f"Error computing related papers: {rel_err}")

    else:
        # Idle state with dataset exploration
        st.info("💡 Enter a research topic above or click one of the sample chips to discover papers.")
        st.markdown("### 🏛️ Indexed Landmark Papers")
        cat_cols = st.columns(len(all_categories))
        for col, cat in zip(cat_cols, all_categories):
            cat_papers = papers[papers["category"] == cat]
            with col:
                st.markdown(f"**{cat}** ({len(cat_papers)})")
                for _, cp in cat_papers.head(3).iterrows():
                    st.caption(f"• {cp['title'][:45]}...")

# ---------------------------------------------------------------------------
# Tab 2: Saved Reading List
# ---------------------------------------------------------------------------

with tab_reading_list:
    st.markdown("### 📑 Your Saved Reading List")

    saved_ids = list(st.session_state.reading_list)
    saved_papers = papers[papers["id"].isin(saved_ids)].reset_index(drop=True)

    if saved_papers.empty:
        st.info(
            "Your reading list is currently empty. Use the **📌 Save** button "
            "on search result cards to bookmark papers for future reference."
        )
    else:
        # Export Actions for reading list
        exp_col1, exp_col2, exp_col3 = st.columns([1.5, 1.5, 4])
        exp_col1.download_button(
            "📥 Download List as CSV",
            data=saved_papers.to_csv(index=False),
            file_name="reading_list.csv",
            mime="text/csv",
            use_container_width=True,
        )
        exp_col2.download_button(
            "📥 Download List as BibTeX",
            data=export_to_bibtex(saved_papers),
            file_name="reading_list.bib",
            mime="text/plain",
            use_container_width=True,
        )
        if exp_col3.button("🗑️ Clear Entire List"):
            st.session_state.reading_list.clear()
            st.rerun()

        st.markdown("---")
        for idx, paper in saved_papers.iterrows():
            pid = paper["id"]
            with st.container(border=True):
                r_title_col, r_btn_col = st.columns([4, 1])
                with r_title_col:
                    st.markdown(f"#### {paper['title']}")
                    st.caption(f"{paper['category']} · {int(paper['year']) if pd.notna(paper['year']) else 'N/A'}")
                    st.write(paper["abstract"])
                with r_btn_col:
                    if paper["url"]:
                        st.link_button("Source ↗", paper["url"], use_container_width=True)
                    if st.button("Remove", key=f"rm_saved_{pid}_{idx}", use_container_width=True):
                        st.session_state.reading_list.remove(pid)
                        st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Paper Comparison
# ---------------------------------------------------------------------------

with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Paper Comparison")

    compared_ids = st.session_state.comparison_list
    compared_papers = papers[papers["id"].isin(compared_ids)].reset_index(drop=True)

    if len(compared_papers) < 2:
        st.info(
            "Select 2 or 3 papers using the **⚖️ Compare** button on search cards "
            "to inspect and compare their metadata and abstracts side-by-side."
        )
    else:
        if st.button("Clear Comparison Tray"):
            st.session_state.comparison_list.clear()
            st.rerun()

        cols = st.columns(len(compared_papers))
        for col, (_, paper) in zip(cols, compared_papers.iterrows()):
            pid = paper["id"]
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {paper['title']}")
                    st.markdown(f"**Research Category:** {paper['category']}")
                    st.markdown(f"**Year:** {int(paper['year']) if pd.notna(paper['year']) else 'N/A'}")
                    word_count = len(paper["abstract"].split())
                    st.markdown(f"**Abstract Length:** {word_count} words")
                    st.markdown("**Abstract:**")
                    st.write(paper["abstract"])
                    if paper["url"]:
                        st.link_button("Read Source ↗", paper["url"], use_container_width=True)
                    if st.button("Remove from compare", key=f"rm_cmp_{pid}", use_container_width=True):
                        st.session_state.comparison_list.remove(pid)
                        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "ResearchMatch ML · Academic Information Retrieval Platform · "
    "Features: TF-IDF, Sentence-Transformers, Reciprocal Rank Fusion · "
    "Dataset: 12 seminal landmark research papers"
)
