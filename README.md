# ResearchMatch ML 📚

A content-based and semantic research paper recommendation system using
Natural Language Processing (NLP), TF-IDF, sentence embeddings, and reciprocal rank fusion.

## Features

- Search research papers by topic or research interest
- Rank papers by textual and semantic similarity
- Filter recommendations by research category
- View paper abstracts, publication years, and source links
- Interactive Streamlit interface
- Automated tests using pytest

## Tech Stack

- Python 3.12+
- Pandas
- Scikit-learn
- Sentence-Transformers
- PyTorch
- Streamlit
- Pytest & Pytest-cov

## Getting Started

### Prerequisites

- Python 3.12 (verified on Python 3.12.10)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/MADHURA1907/researchmatch-ml.git
cd researchmatch-ml
```

### 2. Create and activate a virtual environment

Create a Python 3.12 virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install CPU-only PyTorch (recommended on Windows for a lightweight footprint without bulky CUDA binaries):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install development and test dependencies (includes all production packages):

```powershell
python -m pip install -r requirements-dev.txt
```

> **Note for production:** For production deployments where testing tools are not needed, install with `python -m pip install -r requirements.txt`.

### 4. Run the application

```bash
python -m streamlit run app.py
```

### 5. Run the tests

Run fast offline tests (deselects tests requiring live internet model downloads):

```bash
python -m pytest tests/ -v -k "not test_real_semantic_model_smoke"
```

Run the full test suite (when model weights are cached or internet access is available):

```bash
python -m pytest tests/ -v
```

## How It Works

1. Research paper titles, abstracts, and categories are combined.
2. TF-IDF and dense Sentence Transformers convert text into numerical representations.
3. Cosine similarity compares the search query with each paper.
4. Reciprocal Rank Fusion (RRF) combines lexical and semantic signals to produce recommendations.

## Dataset

The project currently uses a curated demonstration dataset of 12 research papers (`data/papers.csv`).
It is intended as a demonstration dataset, not a comprehensive research index.

## Limitations

Similarity scores indicate vocabulary and semantic overlap, and should not be interpreted
as paper quality or citation impact.

## Future Improvements

- Expand the research-paper dataset
- Add side-by-side paper comparisons
- Add BibTeX citation exports and persistent reading lists
- Evaluate ranking quality on labeled queries

## License

A license has not yet been selected.

## Deployment

ResearchMatch can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open https://share.streamlit.io/
3. Sign in with GitHub and authorize access to the repository.
4. Create a new app and select `MADHURA1907/researchmatch-ml`.
5. Set the main file path to `app.py`.
6. Deploy and test the public application.

The app uses the repository's `requirements.txt` to install its
Python dependencies.

## Project Health Check

Run the sanity check locally before publishing changes:

```bash
python check_project.py
```

Run automated tests:

```bash
# Offline unit tests
python -m pytest tests/ -v -k "not test_real_semantic_model_smoke"

# Full test suite
python -m pytest tests/ -v
```
