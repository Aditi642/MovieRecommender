import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 50
TOP_K = 10
MIN_RATING = 4.0
MAX_USERS = 100

# Hybrid weights
COLLAB_WEIGHT = 0.50
CONTENT_WEIGHT = 0.30
QUALITY_WEIGHT = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

ratings = pd.read_csv(
    "data/ratings.csv"
)

movies = pd.read_csv(
    "data/movies_features.csv"
)

master_movies = pd.read_csv(
    "data/master_movies.csv"
)

print("Ratings:", len(ratings))
print("Movies:", len(movies))


# ============================================================
# KEEP ONLY MOVIES WITH FEATURES
# ============================================================

movies = movies.drop_duplicates(
    subset="movieId"
).copy()

movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
)

movie_ids = set(
    movies["movieId"]
)

ratings = ratings[
    ratings["movieId"].isin(movie_ids)
].copy()

print(
    "Movies used:",
    len(movies)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

test_indices = []

for user_id, user_ratings in ratings.groupby("userId"):

    if len(user_ratings) < 5:
        continue

    # Last rating becomes test rating
    test_indices.append(
        user_ratings.index[-1]
    )

test_ratings = ratings.loc[
    test_indices
].copy()

train_ratings = ratings.drop(
    test_indices
).copy()

print(
    "Training ratings:",
    len(train_ratings)
)

print(
    "Test ratings:",
    len(test_ratings)
)


# ============================================================
# TF-IDF CONTENT MODEL
# ============================================================

print("\nBuilding TF-IDF...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

print(
    "TF-IDF shape:",
    tfidf_matrix.shape
)


# ============================================================
# MOVIE INDEX
# ============================================================

movie_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).drop_duplicates()


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

print("\nBuilding collaborative model...")

user_movie_matrix = train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print(
    "Training matrix:",
    user_movie_matrix.shape
)


# User means

user_means = user_movie_matrix.mean(
    axis=1
)


# Center ratings

centered_matrix = (
    user_movie_matrix
    .sub(user_means, axis=0)
    .fillna(0)
)


print("Training SVD...")

svd = TruncatedSVD(
    n_components=N_COMPONENTS,
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


# Predicted centered ratings

predicted_centered = (
    user_factors @ movie_factors
)


# Add user mean

predicted_ratings = (
    predicted_centered
    + user_means.values.reshape(-1, 1)
)


predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


# ============================================================
# QUALITY SCORE
# ============================================================

print(
    "\nBuilding movie quality scores..."
)

quality_data = master_movies[
    [
        "movieId",
        "vote_average",
        "vote_count"
    ]
].copy()


quality_data["vote_average"] = pd.to_numeric(
    quality_data["vote_average"],
    errors="coerce"
).fillna(0)


quality_data["vote_count"] = pd.to_numeric(
    quality_data["vote_count"],
    errors="coerce"
).fillna(0)


# Normalize rating

quality_data["rating_norm"] = (
    quality_data["vote_average"] / 10
)


# Log transform vote count

quality_data["popularity_norm"] = (
    np.log1p(
        quality_data["vote_count"]
    )
)


max_popularity = (
    quality_data["popularity_norm"].max()
)

if max_popularity > 0:

    quality_data["popularity_norm"] = (
        quality_data["popularity_norm"]
        / max_popularity
    )


# Combine quality

quality_data["quality"] = (
    0.7 * quality_data["rating_norm"]
    +
    0.3 * quality_data["popularity_norm"]
)


quality_scores = pd.Series(
    quality_data["quality"].values,
    index=quality_data["movieId"]
)

print("Quality scores created.")


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(scores)

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(
    user_id,
    top_k=10
):

    if user_id not in predicted_ratings.index:

        return []

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    collab_series = (
        predicted_ratings.loc[user_id]
    )

    collab_movie_ids = (
        collab_series.index
    )

    collab_scores = (
        collab_series.values
    )

    collab_norm = normalize_scores(
        collab_scores
    )


    collab_dict = dict(
        zip(
            collab_movie_ids,
            collab_norm
        )
    )


    # --------------------------------------------------------
    # Movies rated by user
    # --------------------------------------------------------

    seen_movies = set(
        train_ratings[
            train_ratings["userId"] == user_id
        ]["movieId"]
    )


    # --------------------------------------------------------
    # User content profile
    # --------------------------------------------------------

    liked = train_ratings[
        (train_ratings["userId"] == user_id)
        &
        (train_ratings["rating"] >= MIN_RATING)
    ]


    profile_indices = []

    for movie_id in liked["movieId"]:

        if movie_id in movie_index:

            profile_indices.append(
                movie_index[movie_id]
            )


    # If user has no liked movies,
    # use zero content scores.

    if profile_indices:

        user_profile = (
            tfidf_matrix[
                profile_indices
            ].mean(axis=0)
        )

        user_profile = np.asarray(
            user_profile
        )

        content_scores = (
            cosine_similarity(
                user_profile,
                tfidf_matrix
            ).flatten()
        )

    else:

        content_scores = np.zeros(
            len(movies)
        )


    content_norm = normalize_scores(
        content_scores
    )


    # --------------------------------------------------------
    # Build candidate scores
    # --------------------------------------------------------

    candidates = []

    for idx, movie_id in enumerate(
        movies["movieId"]
    ):

        # Don't recommend movies already seen
        if movie_id in seen_movies:
            continue


        collaborative = (
            collab_dict.get(
                movie_id,
                0
            )
        )


        content = (
            content_norm[idx]
        )


        quality = (
            quality_scores.get(
                movie_id,
                0
            )
        )


        hybrid = (
            COLLAB_WEIGHT * collaborative
            +
            CONTENT_WEIGHT * content
            +
            QUALITY_WEIGHT * quality
        )


        candidates.append(
            (
                movie_id,
                hybrid,
                collaborative,
                content,
                quality
            )
        )


    # Sort

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return candidates[:top_k]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    recommendations,
    relevant_movie
):

    recommended_ids = [
        item[0]
        for item in recommendations
    ]

    hits = int(
        relevant_movie
        in recommended_ids
    )

    return hits / TOP_K


def recall_at_k(
    recommendations,
    relevant_movies
):

    if len(relevant_movies) == 0:
        return 0

    recommended_ids = {
        item[0]
        for item in recommendations
    }

    hits = len(
        recommended_ids
        &
        set(relevant_movies)
    )

    return hits / len(
        relevant_movies
    )


def hit_rate(
    recommendations,
    relevant_movie
):

    recommended_ids = [
        item[0]
        for item in recommendations
    ]

    return int(
        relevant_movie
        in recommended_ids
    )


def ndcg_at_k(
    recommendations,
    relevant_movies
):

    relevant_movies = set(
        relevant_movies
    )

    dcg = 0.0

    for position, item in enumerate(
        recommendations,
        start=1
    ):

        movie_id = item[0]

        if movie_id in relevant_movies:

            dcg += (
                1
                /
                np.log2(position + 1)
            )


    ideal_hits = min(
        len(relevant_movies),
        TOP_K
    )

    if ideal_hits == 0:
        return 0


    idcg = sum(
        1
        /
        np.log2(position + 1)
        for position
        in range(
            1,
            ideal_hits + 1
        )
    )


    return dcg / idcg


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating hybrid model..."
)

users = (
    test_ratings["userId"]
    .unique()
)

users = users[
    :MAX_USERS
]


precision_scores = []
recall_scores = []
hit_scores = []
ndcg_scores = []


# For RMSE / MAE

actual_ratings = []
predicted_values = []


evaluated_users = 0


for user_id in users:

    user_test = test_ratings[
        test_ratings["userId"] == user_id
    ]

    if len(user_test) == 0:
        continue


    # --------------------------------------------------------
    # Test movie
    # --------------------------------------------------------

    test_movie = user_test.iloc[
        -1
    ]["movieId"]

    actual_rating = user_test.iloc[
        -1
    ]["rating"]


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = (
        get_recommendations(
            user_id,
            TOP_K
        )
    )


    if not recommendations:
        continue


    # Relevant movies in test set
    relevant_movies = (
        user_test[
            user_test["rating"] >= MIN_RATING
        ]["movieId"]
        .tolist()
    )


    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    precision_scores.append(
        precision_at_k(
            recommendations,
            test_movie
        )
    )


    recall_scores.append(
        recall_at_k(
            recommendations,
            relevant_movies
        )
    )


    hit_scores.append(
        hit_rate(
            recommendations,
            test_movie
        )
    )


    ndcg_scores.append(
        ndcg_at_k(
            recommendations,
            relevant_movies
        )
    )


    # --------------------------------------------------------
    # Predicted rating
    # --------------------------------------------------------

    if (
        user_id in predicted_ratings.index
        and
        test_movie
        in predicted_ratings.columns
    ):

        prediction = (
            predicted_ratings.loc[
                user_id,
                test_movie
            ]
        )

        actual_ratings.append(
            actual_rating
        )

        predicted_values.append(
            prediction
        )


    evaluated_users += 1


# ============================================================
# RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "HYBRID RECOMMENDATION EVALUATION"
)

print(
    "=========================================="
)


if evaluated_users > 0:

    print(
        f"Users evaluated: "
        f"{evaluated_users}"
    )


    print(
        f"\nPrecision@{TOP_K}: "
        f"{np.mean(precision_scores):.4f}"
    )


    print(
        f"Recall@{TOP_K}: "
        f"{np.mean(recall_scores):.4f}"
    )


    print(
        f"Hit Rate@{TOP_K}: "
        f"{np.mean(hit_scores):.4f}"
    )


    print(
        f"NDCG@{TOP_K}: "
        f"{np.mean(ndcg_scores):.4f}"
    )


# ============================================================
# RMSE / MAE
# ============================================================

if len(actual_ratings) > 0:

    rmse = np.sqrt(
        mean_squared_error(
            actual_ratings,
            predicted_values
        )
    )

    mae = mean_absolute_error(
        actual_ratings,
        predicted_values
    )


    print(
        f"\nRMSE: {rmse:.4f}"
    )

    print(
        f"MAE: {mae:.4f}"
    )


print(
    "\n=========================================="
)

print("Evaluation complete.")

print(
    "=========================================="
)
