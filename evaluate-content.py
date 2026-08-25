import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD DATA
# ==========================================

movies = pd.read_csv(
    "data/movies_features.csv"
)

ratings = pd.read_csv(
    "data/ratings.csv"
)


# ==========================================
# CLEAN
# ==========================================

movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
)


# ==========================================
# TF-IDF
# ==========================================

print("Building TF-IDF...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)


# ==========================================
# MOVIE INDEX
# ==========================================

movie_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).drop_duplicates()


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(
    min_rating=4.0,
    top_k=10,
    max_users=100
):

    hits = 0
    evaluated_users = 0

    # Users with enough ratings
    user_counts = ratings.groupby(
        "userId"
    ).size()

    eligible_users = user_counts[
        user_counts >= 5
    ].index

    eligible_users = eligible_users[
        :max_users
    ]

    print(
        f"\nEvaluating {len(eligible_users)} users..."
    )

    for user_id in eligible_users:

        user_ratings = ratings[
            ratings["userId"] == user_id
        ]

        # Movies this user liked
        liked_movies = user_ratings[
            user_ratings["rating"] >= min_rating
        ]

        if len(liked_movies) < 2:
            continue

        # Hold out one liked movie
        test_movie = liked_movies.iloc[
            -1
        ]["movieId"]

        if test_movie not in movie_index:
            continue

        # Movies the user already rated
        seen_movies = set(
            user_ratings["movieId"]
        )

        # Build user's content profile
        train_movies = liked_movies[
            liked_movies["movieId"] != test_movie
        ]

        profile_indices = []

        for movie_id in train_movies["movieId"]:

            if movie_id in movie_index:

                profile_indices.append(
                    movie_index[movie_id]
                )

        if not profile_indices:
            continue

        # Average liked movie vectors
        user_profile = (
            tfidf_matrix[
                profile_indices
            ].mean(axis=0)
        )

        # Compare profile against all movies
        user_profile = np.asarray(user_profile)

        scores = cosine_similarity(
            user_profile,
            tfidf_matrix
        ).flatten()

        # Don't recommend seen movies
        candidate_scores = []

        for idx, score in enumerate(scores):

            movie_id = movies.iloc[
                idx
            ]["movieId"]

            if movie_id not in seen_movies:

                candidate_scores.append(
                    (idx, score)
                )

        # Sort
        candidate_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_movies = candidate_scores[
            :top_k
        ]

        recommended_ids = [
            movies.iloc[idx]["movieId"]
            for idx, score in top_movies
        ]

        # Check hit
        if test_movie in recommended_ids:

            hits += 1

        evaluated_users += 1

    # ======================================
    # RESULT
    # ======================================

    if evaluated_users == 0:

        print(
            "No users could be evaluated."
        )

        return

    hit_rate = (
        hits / evaluated_users
    )

    print("\n===== EVALUATION RESULTS =====")

    print(
        "Users evaluated:",
        evaluated_users
    )

    print(
        "Hits:",
        hits
    )

    print(
        f"Hit Rate@{top_k}: "
        f"{hit_rate:.4f}"
    )


# ==========================================
# RUN
# ==========================================

evaluate_model(
    min_rating=4.0,
    top_k=10,
    max_users=100
)