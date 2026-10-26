
import streamlit as st

from src.recommender import (
    load_papers,
    build_recommender,
    recommend_papers,
)

st.set_page_config(
    page_title="ResearchMatch ML",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ResearchMatch ML")
st.markdown(
    "Discover relevant research papers using **NLP, TF-IDF, "
    "and cosine similarity**."
)

@st.cache_data
def get_papers():
    return load_papers()

@st.cache_resource
def get_model():
    papers = get_papers()
    vectorizer, matrix = build_recommender(papers)
    return vectorizer, matrix

try:
    papers = get_papers()
    vectorizer, matrix = get_model()
except Exception as exc:
    st.error(f"Unable to load the research dataset: {exc}")
    st.stop()

# Sidebar
st.sidebar.header("Explore")
st.sidebar.write(f"📄 {len(papers)} papers available")

categories = ["All categories"] + sorted(
    papers["category"].dropna().unique().tolist()
)
selected_category = st.sidebar.selectbox(
    "Research area",
    categories,
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5,
)

# Search
query = st.text_input(
    "What would you like to research?",
    placeholder="e.g. computer vision for medical imaging",
)

if query.strip():
    results = recommend_papers(
        query,
        papers,
        vectorizer,
        matrix,
        top_k=len(papers),
    )

    if selected_category != "All categories":
        results = results[
            results["category"] == selected_category
        ]

    results = results.head(top_k)

    st.subheader("Recommended papers")
    st.caption(
        f"{len(results)} results ranked by textual similarity"
    )

    if results.empty:
        st.info(
            "No matching papers found. Try another topic "
            "or select a different category."
        )

    for _, paper in results.iterrows():
        with st.container(border=True):
            st.markdown(f"### {paper['title']}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(paper["abstract"])
                st.caption(
                    f"{paper['category']} · "
                    f"{int(paper['year']) if paper['year'] == paper['year'] else 'Year unknown'}"
                )

            with col2:
                st.metric(
                    "Text similarity",
                    f"{paper['similarity']:.1%}",
                )

                if paper["url"]:
                    st.link_button(
                        "Read paper ↗",
                        paper["url"],
                    )

else:
    st.info(
        "Enter a research topic above to find relevant papers."
    )

    st.subheader("Try a sample topic")

    examples = [
        "transformer language models",
        "computer vision image classification",
        "generative AI and retrieval augmented generation",
        "deep learning optimization",
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            st.write(f"• {example}")

st.divider()
st.caption(
    "ResearchMatch ML · Content-based recommendations · "
    "Scores represent textual similarity, not paper quality."
)
