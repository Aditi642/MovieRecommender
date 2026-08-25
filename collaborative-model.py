import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD


# ==========================================
# LOAD RATINGS
# ==========================================

ratings = pd.read_csv(
    "data/ratings.csv"
)

print("Ratings:", len(ratings))
print("Users:", ratings["userId"].nunique())
print("Movies:", ratings["movieId"].nunique())


# ==========================================
# CREATE USER-MOVIE MATRIX
# ==========================================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0
)

print(
    "\nUser-movie matrix:",
    user_movie_matrix.shape
)


# ==========================================
# MATRIX FACTORIZATION
# ==========================================

print("\nTraining SVD model...")

n_components = 50

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd.fit_transform(
    user_movie_matrix
)

movie_factors = svd.components_


print(
    "User factors:",
    user_factors.shape
)

print(
    "Movie factors:",
    movie_factors.shape
)


# ==========================================
# PREDICT RATINGS
# ==========================================

predicted_ratings = (
    user_factors @ movie_factors
)

predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


# ==========================================
# RECOMMEND MOVIES
# ==========================================

movies = pd.read_csv(
    "data/master_movies.csv"
)

movie_titles = pd.Series(
    movies["title"].values,
    index=movies["movieId"]
)


def recommend_for_user(
    user_id,
    n=10
):

    if user_id not in predicted_ratings.index:

        print(
            f"User {user_id} not found."
        )

        return

    # Predicted scores
    scores = predicted_ratings.loc[
        user_id
    ]

    # Movies already rated
    watched = ratings[
        ratings["userId"] == user_id
    ]["movieId"]

    watched = set(watched)

    # Remove watched movies
    scores = scores[
        ~scores.index.isin(watched)
    ]

    # Top recommendations
    recommendations = scores.sort_values(
        ascending=False
    ).head(n)

    print(
        f"\n===== RECOMMENDATIONS FOR USER "
        f"{user_id} ====="
    )

    for movie_id, score in recommendations.items():

        title = movie_titles.get(
            movie_id,
            "Unknown Movie"
        )

        print(
            f"{title} "
            f"(predicted rating: {score:.2f})"
        )


# ==========================================
# TEST
# ==========================================

for user_id in [1, 10, 50, 100, 200]:

    recommend_for_user(
        user_id=user_id,
        n=10
    )