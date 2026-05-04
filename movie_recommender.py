import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
EXTRACT_DIR = DATA_DIR / "ml-latest-small"
MOVIES_CSV = EXTRACT_DIR / "movies.csv"
RATINGS_CSV = EXTRACT_DIR / "ratings.csv"
TAGS_CSV = EXTRACT_DIR / "tags.csv"
BOLLYWOOD_CSV = DATA_DIR / "bollywood.csv"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")


def download_and_extract_data(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MOVIES_CSV.exists() and not force:
        return

    print(f"Downloading MovieLens dataset to {ZIP_PATH}...")
    response = requests.get(DATA_URL, stream=True, timeout=30)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Data download and extraction complete.")


def load_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    download_and_extract_data(force=force)
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    tags = pd.read_csv(TAGS_CSV)
    return movies, ratings, tags


def download_bollywood_data(force: bool = False) -> pd.DataFrame:
    if BOLLYWOOD_CSV.exists() and not force:
        return pd.read_csv(BOLLYWOOD_CSV)

    print("Downloading Bollywood movies from Kaggle...")
    try:
        import kaggle
        # Check if authenticated
        kaggle.api.authenticate()
    except Exception as e:
        raise FileNotFoundError(
            f"Kaggle authentication failed: {e}. "
            "Please set up Kaggle API credentials as described at: "
            "https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    # Download the Bollywood Movies Dataset
    kaggle.api.dataset_download_files('dineshpiyasamara/bollywood-movies-dataset', path=str(DATA_DIR), unzip=True)

    # Find the downloaded file
    bollywood_files = list(DATA_DIR.glob("bollywood*.csv"))
    if not bollywood_files:
        # Try alternative file patterns
        csv_files = list(DATA_DIR.glob("*.csv"))
        bollywood_files = [f for f in csv_files if 'bollywood' in f.name.lower() or 'movie' in f.name.lower()]

    if not bollywood_files:
        raise FileNotFoundError("Bollywood dataset file not found after download")

    bollywood_path = bollywood_files[0]
    bollywood = pd.read_csv(bollywood_path)

    # Rename columns to match our expected format
    column_mapping = {
        'movie_id': 'movieId',
        'title': 'title',
        'genre': 'genres',
        'overview': 'overview'
    }
    bollywood = bollywood.rename(columns=column_mapping)

    # Ensure required columns exist
    if 'movieId' not in bollywood.columns:
        bollywood['movieId'] = range(1, len(bollywood) + 1)
    if 'genres' not in bollywood.columns:
        bollywood['genres'] = ''
    if 'overview' not in bollywood.columns:
        bollywood['overview'] = ''

    bollywood = bollywood[['movieId', 'title', 'genres', 'overview']].dropna(subset=['title'])
    bollywood.to_csv(BOLLYWOOD_CSV, index=False)
    return bollywood


def load_bollywood_data(force: bool = False) -> pd.DataFrame:
    if BOLLYWOOD_CSV.exists() and not force:
        return pd.read_csv(BOLLYWOOD_CSV)

    try:
        return download_bollywood_data(force=force)
    except Exception as e:
        raise FileNotFoundError(
            f"Bollywood dataset not found locally and download failed: {e}. "
            "Please set up Kaggle API credentials or add data/bollywood.csv manually."
        )


def normalize_title(title: str) -> str:
    normalized = title.strip().lower()
    normalized = re.sub(r"\s*\(\d{4}\)$", "", normalized)
    comma_match = re.match(r"^(.*),\s*(the|a|an)$", normalized)
    if comma_match:
        normalized = f"{comma_match.group(2)} {comma_match.group(1)}"
    return normalized


def preprocess_text(text: str) -> str:
    """Advanced text preprocessing for better similarity matching."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common stop words that don't add value for movie similarity
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)


def clean_metadata(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    movies = movies.copy()
    movies["title_clean"] = movies["title"].apply(normalize_title)

    if "genres" in movies:
        movies["genres"] = movies["genres"].replace("(no genres listed)", "", regex=False)
        movies["genres_clean"] = movies["genres"].str.replace(r"\|", " ", regex=True).str.lower().str.strip()
    else:
        movies["genres"] = ""
        movies["genres_clean"] = ""

    if tags is not None and "tag" in tags.columns:
        tags_agg = (
            tags.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)))
            .reset_index()
        )
        movies = movies.merge(tags_agg, on="movieId", how="left")
        movies["tag"] = movies["tag"].fillna("")
    else:
        movies["tag"] = ""

    if "overview" in movies.columns:
        overview_text = movies["overview"].fillna("").str.lower()
    else:
        overview_text = ""

    weighted_title = movies["title_clean"].fillna("")
    weighted_genres = movies["genres_clean"].fillna("")
    movies["metadata"] = (
        weighted_title + " " + weighted_title + " " + weighted_title + " "
        + weighted_genres + " " + weighted_genres + " "
        + movies["tag"].fillna("") + " "
        + overview_text
    )
    # Apply advanced text preprocessing for better similarity
    movies["metadata"] = movies["metadata"].apply(preprocess_text)
    movies = movies.reset_index(drop=True)
    return movies


def prepare_recommender(movies: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray, dict[str, int]]:
    # Enhanced TF-IDF with tuned parameters for better movie similarity
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Use unigrams and bigrams for a more stable representation
        min_df=2,  # Ignore extremely rare terms that are unlikely to generalize
        max_df=0.90,  # Remove very common terms across movies
        sublinear_tf=True,
        use_idf=True,
        norm='l2',
        max_features=6000,
    )
    tfidf_matrix = vectorizer.fit_transform(movies["metadata"])

    # Use cosine similarity and smooth scores for ranking
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity = (similarity + 0.01) / 1.01

    index_mapping = {title: idx for idx, title in enumerate(movies["title_clean"])}
    return vectorizer, similarity, index_mapping


def get_imdb_rating(movie_title: str, movies: pd.DataFrame) -> str:
    """Get IMDb rating for a movie using TMDB API."""
    if not TMDB_API_KEY:
        return "N/A"

    # Find the movie in our dataset
    movie_row = movies[movies["title_clean"] == normalize_title(movie_title)]
    if movie_row.empty:
        return "N/A"

    # Try to get TMDB ID if available
    tmdb_id = None
    if "movieId" in movie_row.columns:
        # For MovieLens, we might need to search by title
        title = movie_row["title"].iloc[0]
        # Remove year from title for search
        search_title = re.sub(r"\s*\(\d{4}\)$", "", title)

        try:
            # Search for the movie on TMDB
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "query": search_title,
                "year": re.search(r"\((\d{4})\)$", title).group(1) if re.search(r"\((\d{4})\)$", title) else None
            }
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])

            if results:
                tmdb_id = results[0]["id"]
        except Exception:
            pass

    if tmdb_id:
        try:
            # Get movie details including IMDb rating
            details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            params = {"api_key": TMDB_API_KEY}
            response = requests.get(details_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            imdb_id = data.get("imdb_id")
            vote_average = data.get("vote_average")
            vote_count = data.get("vote_count")

            if vote_average and vote_count and vote_count > 10:
                return f"{vote_average:.1f}"
            elif imdb_id:
                return f"IMDb: {imdb_id}"
        except Exception:
            pass

    return "N/A"


def find_best_title(query: str, movies: pd.DataFrame, index_mapping: dict[str, int]) -> str:
    query_clean = normalize_title(query)
    if query_clean in index_mapping:
        return query_clean

    contains = [title for title in index_mapping if query_clean in title]
    if contains:
        return contains[0]

    suggested = sorted(index_mapping.keys(), key=lambda title: (len(title), title))
    for title in suggested:
        if title.startswith(query_clean) or query_clean.startswith(title):
            return title

    raise ValueError(f"Title not found in dataset: {query}")


def recommend(movie_title: str, movies: pd.DataFrame, similarity: np.ndarray, index_mapping: dict[str, int], top_n: int = 10, include_imdb: bool = True) -> pd.DataFrame:
    movie_title_key = find_best_title(movie_title, movies, index_mapping)
    idx = index_mapping[movie_title_key]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1 : top_n + 1]]
    recommendations = movies.iloc[top_indices][["title", "genres"]].copy()
    recommendations["score"] = [similarity[idx, i] for i in top_indices]

    # Add IMDb ratings to recommendations only if requested
    if include_imdb:
        recommendations["imdb_rating"] = recommendations["title"].apply(lambda title: get_imdb_rating(title, movies))

    return recommendations.reset_index(drop=True)


def get_top_genres(movies: pd.DataFrame, top_n: int = 8) -> list[dict[str, int]]:
    genre_counts = movies["genres"].str.split("|", expand=True).stack().value_counts().head(top_n)
    return [{"name": genre, "count": int(count)} for genre, count in genre_counts.items()]


def get_top_rated_movies(movies: pd.DataFrame, ratings: pd.DataFrame, min_ratings: int = 50, top_n: int = 5) -> list[dict[str, str]]:
    rating_summary = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"), count=("rating", "count")).reset_index()
    rating_summary = rating_summary[rating_summary["count"] >= min_ratings].sort_values(["avg_rating", "count"], ascending=[False, False]).head(top_n)
    top_rated = movies.merge(rating_summary, on="movieId")[['title', 'genres']].copy()
    return top_rated.to_dict("records")


def get_dataset_summary(movies: pd.DataFrame, ratings: pd.DataFrame | None = None, tags: pd.DataFrame | None = None) -> dict:
    summary = {
        "num_movies": int(len(movies)),
        "top_genres": get_top_genres(movies),
    }
    if ratings is not None:
        summary["num_ratings"] = int(len(ratings))
        summary["top_rated"] = get_top_rated_movies(movies, ratings)
    else:
        summary["num_ratings"] = 0
        summary["top_rated"] = []
    if tags is not None:
        summary["num_tags"] = int(len(tags))
    else:
        summary["num_tags"] = 0
    return summary


def compute_accuracy_percent(movies: pd.DataFrame, ratings: pd.DataFrame, similarity: np.ndarray, index_mapping: dict[str, int], top_n: int = 10, sample_users: int = 120) -> float:
    liked = ratings[ratings["rating"] >= 4.0]
    good_users = liked["userId"].value_counts()
    good_users = good_users[good_users >= 5].index.tolist()
    if not good_users:
        return 0.0

    rng = np.random.default_rng(42)
    sample_ids = rng.choice(good_users, min(sample_users, len(good_users)), replace=False)
    precisions = []

    for user_id in sample_ids:
        liked_movies = liked[liked["userId"] == user_id]["movieId"].unique()
        if len(liked_movies) == 0:
            continue

        query_movie_id = liked_movies[0]
        query_title_row = movies[movies["movieId"] == query_movie_id]
        if query_title_row.empty:
            continue

        query_title = query_title_row["title"].iloc[0]
        try:
            recs = recommend(query_title, movies, similarity, index_mapping, top_n=top_n, include_imdb=False)
        except ValueError:
            continue

        liked_titles = set(movies[movies["movieId"].isin(liked_movies)]["title"].tolist())
        hits = sum(1 for title in recs["title"] if title in liked_titles)
        precisions.append(hits / top_n)

    if not precisions:
        return 0.0

    return round(float(np.mean(precisions) * 100), 1)


def print_eda(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> None:
    print("\nEDA: dataset summary")
    print(f"Movies: {len(movies)}")
    print(f"Ratings: {len(ratings)}")
    print(f"Tags: {len(tags)}")

    print("\nTop 10 movie genres by count:")
    genre_counts = movies["genres"].str.split("|", expand=True).stack().value_counts().head(10)
    print(genre_counts.to_string())

    print("\nTop 10 movies by average rating (min 50 ratings):")
    rating_summary = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"), count=("rating", "count")).reset_index()
    rating_summary = rating_summary[rating_summary["count"] >= 50].sort_values(["avg_rating", "count"], ascending=[False, False]).head(10)
    top_rated = movies.merge(rating_summary, on="movieId")[['title', 'genres', 'avg_rating', 'count']]
    for _, row in top_rated.iterrows():
        print(f"  {row['title']} | {row['genres']} | avg={row['avg_rating']:.3f} | count={int(row['count'])}")

    print("\nExample movie tags and metadata cleaning samples:")
    example = movies.loc[movies["tag"] != "", ["title", "genres", "tag", "metadata"]].head(5)
    for _, row in example.iterrows():
        print(f"  {row['title']} | {row['genres']} | tag={row['tag']} | metadata={row['metadata']}")


def print_recommendation_examples(movies: pd.DataFrame, similarity: np.ndarray, index_mapping: dict[str, int]) -> None:
    print("\nRecommendation examples:")
    seeds = ["Toy Story", "The Godfather", "Pulp Fiction", "Mean Girls", "La La Land"]
    for seed in seeds:
        try:
            recommendations = recommend(seed, movies, similarity, index_mapping, top_n=5)
            print(f"\nSeed movie: {seed}")
            for idx, row in recommendations.iterrows():
                print(f"  {idx+1}. {row['title']} ({row['genres']}) [score={row['score']:.3f}]")
        except ValueError:
            print(f"  Seed movie '{seed}' not found.")


def build_and_run_demo(force: bool = False) -> None:
    movies, ratings, tags = load_data(force=force)
    movies = clean_metadata(movies, tags)
    _, similarity, index_mapping = prepare_recommender(movies)
    print_eda(movies, ratings, tags)
    print_recommendation_examples(movies, similarity, index_mapping)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MovieLens content-based recommender demo")
    parser.add_argument("--query", type=str, help="Movie title to find recommendations for")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of the dataset")
    args = parser.parse_args()

    movies, _, tags = load_data(force=args.force_download)
    movies = clean_metadata(movies, tags)
    _, similarity, index_mapping = prepare_recommender(movies)

    if args.query:
        recommendations = recommend(args.query, movies, similarity, index_mapping, top_n=args.top)
        print(f"Recommendations for '{args.query}':")
        for idx, row in recommendations.iterrows():
            print(f"  {idx+1}. {row['title']} | {row['genres']} | score={row['score']:.3f}")
    else:
        build_and_run_demo(force=args.force_download)


if __name__ == "__main__":
    main()
