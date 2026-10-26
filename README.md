
# ResearchMatch ML 📚

A content-based research paper recommendation system using
Natural Language Processing (NLP), TF-IDF, and cosine similarity.

## Features

- Search research papers by topic or research interest
- Rank papers by textual similarity
- Filter recommendations by research category
- View paper abstracts, publication years, and source links
- Interactive Streamlit interface
- Automated tests using pytest

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MADHURA1907/researchmatch-ml.git
cd researchmatch-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

### 5. Run the tests

```bash
python -m pytest tests/ -v
```

## How It Works

1. Research paper titles, abstracts, and categories are combined.
2. TF-IDF converts text into numerical feature vectors.
3. Cosine similarity compares the search query with each paper.
4. Papers are ranked by their similarity scores.

## Dataset

The project currently uses a small, curated dataset of 12 research papers.
It is intended as a demonstration dataset, not a comprehensive research index.

## Limitations

This is a content-based baseline. It does not understand semantic
similarity as deeply as embedding-based models, and similarity scores
should not be interpreted as paper quality or prediction accuracy.

## Future Improvements

- Expand the research-paper dataset
- Add semantic search using sentence embeddings
- Evaluate ranking quality on labeled queries
- Deploy the app publicly

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

Run the following command locally before publishing changes:

```bash
python check_project.py
```

Run automated tests with:

```bash
python -m pytest tests/ -v
```

