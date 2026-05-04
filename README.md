# MovieLens Content-Based Recommender

This small project downloads the MovieLens `ml-latest-small` dataset, performs metadata cleaning and simple EDA, builds a content-based recommender using text features and cosine similarity, and exposes a Flask endpoint for movie recommendations.

## Files

- `movie_recommender.py`: downloads the dataset, cleans metadata, builds the recommender, and prints EDA/recommendation examples.
- `app.py`: simple Flask endpoint at `/recommend?title=...` returning JSON movie recommendations.
- `requirements.txt`: Python dependencies.

## Setup

From the workspace root:

```bash
/Users/sanjana/movie_predictor/.venv/bin/python -m pip install -r requirements.txt
```

## Run the demo

```bash
/Users/sanjana/movie_predictor/.venv/bin/python movie_recommender.py
```

This will:
- download MovieLens `ml-latest-small`
- load `movies.csv`, `ratings.csv`, and `tags.csv`
- clean metadata by combining title, genres, and tags
- show EDA statistics and recommendation examples

## Query a movie directly

```bash
/Users/sanjana/movie_predictor/.venv/bin/python movie_recommender.py --query "The Godfather"
```

This prints the top content-based recommendations for the supplied movie title.

## Bollywood dataset support

The web app includes a sample Bollywood dataset with 10 popular Indian movies. You can also expand it:

1. The app comes with a sample `data/bollywood.csv` with Bollywood movies.
2. To use a larger Kaggle dataset, set up Kaggle API credentials:

```bash
# Install Kaggle CLI and set up credentials
pip install kaggle
# Follow: https://github.com/Kaggle/kaggle-api#api-credentials

# Then run the app - it will download the full Bollywood Movies Dataset automatically
/Users/sanjana/movie_predictor/.venv/bin/python /Users/sanjana/movie_predictor/app.py
```

If Bollywood data is available, the predictor page will show a dataset selector for `MovieLens` and `Bollywood`.

## Run the Flask API

```bash
/Users/sanjana/movie_predictor/.venv/bin/python app.py
```

Then visit the interactive web app:

```bash
http://127.0.0.1:5000/
```

Available pages:
- Home: `/`
- How It Works: `/how-it-works`
- Real-time Predictor: `/predictor`
- Analysis: `/analysis`
- Accuracy: `/accuracy`

## Example output

The recommender returns similar movies based on text features from movie title, genres, and user tags.
# movie_predictor
