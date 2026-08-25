import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD


# ==========================================
# LOAD DATA
# ==========================================

ratings = pd.read_csv(
    "data/ratings.csv"
)

movies = pd.read_csv(
    "data/master_movies.csv"
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
    values="rating"
)

print(
    "\nUser-movie matrix:",
    user_movie_matrix.shape
)


# ==========================================
# USER MEAN RATINGS
# ==========================================

user_means = user_movie_matrix.mean(
    axis=1
)


# ==========================================
# CENTER RATINGS
# ==========================================

centered_matrix = (
    user_movie_matrix
    .sub(user_means, axis=0)
)


# Missing values are NOT ratings.
# For matrix factorization we temporarily
# represent them as zero AFTER centering.

centered_matrix = centered_matrix.fillna(0)


# ==========================================
# MATRIX FACTORIZATION
# ==========================================

print("\nTraining centered SVD model...")

svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_factors = svd.fit_transform(
    centered_matrix
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
# PREDICT CENTERED RATINGS
# ==========================================

predicted_centered = (
    user_factors @ movie_factors
)


# ==========================================
# ADD USER MEANS BACK
# ==========================================

predicted_ratings = (
    predicted_centered
    + user_means.values.reshape(-1, 1)
)


predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


# ==========================================
# CLIP RATINGS
# ==========================================

predicted_ratings = predicted_ratings.clip(
    lower=0.5,
    upper=5.0
)


# ==========================================
# MOVIE TITLES
# ==========================================

movie_titles = pd.Series(
    movies["title"].values,
    index=movies["movieId"]
)


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_for_user(
    user_id,
    n=10
):

    if user_id not in predicted_ratings.index:

        print(
            f"User {user_id} not found."
        )

        return

    scores = predicted_ratings.loc[
        user_id
    ].copy()

    # Movies already rated
    watched = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

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
# TEST USERS
# ==========================================

for user_id in [1, 10, 50, 100, 200]:

    recommend_for_user(
        user_id=user_id,
        n=10
    )