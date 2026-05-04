from flask import Flask, jsonify, redirect, render_template, request, url_for

from movie_recommender import (
    clean_metadata,
    compute_accuracy_percent,
    get_dataset_summary,
    load_bollywood_data,
    load_data,
    prepare_recommender,
    recommend,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

movies, ratings, tags = load_data()
movies = clean_metadata(movies, tags)
_, similarity, index_mapping = prepare_recommender(movies)
summary = get_dataset_summary(movies, ratings, tags)
accuracy_percent = compute_accuracy_percent(
    movies, ratings, similarity, index_mapping, top_n=10, sample_users=120
)

bollywood_movies = None
bollywood_similarity = None
bollywood_index_mapping = None
bollywood_summary = None

try:
    bollywood_movies = load_bollywood_data()
    bollywood_movies = clean_metadata(bollywood_movies)
    _, bollywood_similarity, bollywood_index_mapping = prepare_recommender(bollywood_movies)
    bollywood_summary = get_dataset_summary(bollywood_movies)
except Exception:
    bollywood_movies = None
    bollywood_summary = None


@app.route("/")
def home():
    return render_template("home.html", title="Home")


@app.route("/home")
def home_redirect():
    return redirect(url_for("home"))


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html", title="How it Works")


@app.route("/predictor")
def predictor():
    datasets = [{"key": "movielens", "label": "MovieLens"}]
    if bollywood_movies is not None:
        datasets.append({"key": "bollywood", "label": "Bollywood"})
    return render_template("predictor.html", title="Predictor", datasets=datasets)


@app.route("/analysis")
def analysis():
    return render_template(
        "analysis.html",
        title="Analysis",
        summary=summary,
        bollywood_summary=bollywood_summary,
    )


@app.route("/accuracy")
def accuracy():
    return render_template("accuracy.html", title="Accuracy", accuracy=accuracy_percent)


@app.route("/recommend")
def recommend_endpoint():
    title = request.args.get("title", "")
    dataset = request.args.get("dataset", "movielens")

    if not title:
        return jsonify({"error": "Please provide a movie title"}), 400

    try:
        if dataset == "bollywood" and bollywood_movies is not None:
            recs = recommend(
                title, bollywood_movies, bollywood_similarity, bollywood_index_mapping
            )
        else:
            recs = recommend(title, movies, similarity, index_mapping)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    return jsonify({"recommendations": recs.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(debug=True, port=5001)