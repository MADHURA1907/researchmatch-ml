
from pathlib import Path

from src.recommender import load_papers, build_recommender


def main():
    root = Path(__file__).resolve().parent
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "data/papers.csv",
        "src/recommender.py",
        "tests/test_recommender.py",
    ]

    missing = [name for name in required_files if not (root / name).is_file()]

    if missing:
        raise SystemExit(f"Missing project files: {missing}")

    papers = load_papers()
    vectorizer, matrix = build_recommender(papers)

    print("ResearchMatch project check passed!")
    print(f"Dataset records: {len(papers)}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print("Required project files: all present")


if __name__ == "__main__":
    main()
