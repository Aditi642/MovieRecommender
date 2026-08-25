import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies_features.csv"

N_COMPONENTS = 50
MAX_FEATURES = 50000
MIN_RATING = 4.0

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

print("Ratings:", len(ratings))
print("Movies:", len(movies))


# ============================================================
# BASIC CLEANING
# ============================================================

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
).copy()

movies = movies.drop_duplicates(
    subset=["movieId"]
).copy()

movies["movieId"] = movies["movieId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)
ratings["userId"] = ratings["userId"].astype(int)


# ============================================================
# CHECK FEATURES
# ============================================================

if "combined_features" not in movies.columns:

    print(
        "\ncombined_features not found."
    )

    print(
        "Creating combined_features from available columns..."
    )

    feature_columns = [
        "genres_clean",
        "keywords_clean",
        "cast_clean",
        "director_clean"
    ]

    available_features = [
        col
        for col in feature_columns
        if col in movies.columns
    ]

    if not available_features:

        print(
            "\nAvailable columns:"
        )

        print(
            movies.columns.tolist()
        )

        raise ValueError(
            "\nCould not find feature columns."
            "\nMake sure feature-engineering.py "
            "has been run first."
        )

    movies["combined_features"] = (
        movies[available_features]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )


movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
    .astype(str)
)


# ============================================================
# KEEP ONLY MOVIES WITH RATINGS
# ============================================================

rated_movie_ids = set(
    ratings["movieId"].unique()
)

movies = movies[
    movies["movieId"].isin(rated_movie_ids)
].copy()

movies = movies.reset_index(drop=True)

print(
    "Movies used:",
    len(movies)
)


# ============================================================
# PER-USER TRAIN / TEST SPLIT
# ============================================================

print(
    "\nCreating per-user train/test split..."
)

train_parts = []
test_parts = []

rng = np.random.RandomState(
    RANDOM_STATE
)

for user_id, group in ratings.groupby(
    "userId"
):

    group = group.copy()

    # Need at least 2 ratings
    if len(group) < 2:

        train_parts.append(group)
        continue

    # Use 80% for training
    test_size = max(
        1,
        int(round(len(group) * 0.20))
    )

    # Never remove all ratings
    test_size = min(
        test_size,
        len(group) - 1
    )

    test_indices = rng.choice(
        group.index,
        size=test_size,
        replace=False
    )

    test_group = group.loc[
        test_indices
    ]

    train_group = group.drop(
        test_indices
    )

    train_parts.append(
        train_group
    )

    test_parts.append(
        test_group
    )


train_ratings = pd.concat(
    train_parts,
    ignore_index=True
)

if test_parts:

    test_ratings = pd.concat(
        test_parts,
        ignore_index=True
    )

else:

    test_ratings = pd.DataFrame(
        columns=ratings.columns
    )


print(
    "Training ratings:",
    len(train_ratings)
)

print(
    "Test ratings:",
    len(test_ratings)
)


# ============================================================
# BUILD MOVIE INDEX
# ============================================================

movie_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).drop_duplicates()


# ============================================================
# BUILD TF-IDF
# ============================================================

print(
    "\nBuilding TF-IDF..."
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=MAX_FEATURES,
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
# BUILD TRAINING USER-MOVIE MATRIX
# ============================================================

print(
    "\nCreating training matrix..."
)

training_movie_ids = sorted(
    train_ratings["movieId"].unique()
)

training_movie_ids = [
    movie_id
    for movie_id in training_movie_ids
    if movie_id in movie_index
]

training_matrix = train_ratings[
    train_ratings["movieId"].isin(
        training_movie_ids
    )
].pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print(
    "Training matrix:",
    training_matrix.shape
)


# ============================================================
# CENTER RATINGS
# ============================================================

user_means = training_matrix.mean(
    axis=1
)

centered_matrix = (
    training_matrix
    .sub(user_means, axis=0)
    .fillna(0)
)


# ============================================================
# TRAIN SVD
# ============================================================

print(
    "\nTraining SVD..."
)

n_components = min(
    N_COMPONENTS,
    centered_matrix.shape[0] - 1,
    centered_matrix.shape[1] - 1
)

svd = TruncatedSVD(
    n_components=n_components,
    random_state=RANDOM_STATE
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


# ============================================================
# CREATE COLLABORATIVE PREDICTION MATRIX
# ============================================================

predicted_centered = (
    user_factors @ movie_factors
)

predicted_ratings = (
    predicted_centered
    + user_means.values.reshape(-1, 1)
)

predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=training_matrix.index,
    columns=training_matrix.columns
)

predicted_ratings = predicted_ratings.clip(
    lower=0.5,
    upper=5.0
)


# ============================================================
# MOVIE QUALITY SCORES
# ============================================================

print(
    "\nBuilding movie quality scores..."
)

# Prefer TMDB vote information if available.
if (
    "vote_average" in movies.columns
    and "vote_count" in movies.columns
):

    vote_average = pd.to_numeric(
        movies["vote_average"],
        errors="coerce"
    ).fillna(0)

    vote_count = pd.to_numeric(
        movies["vote_count"],
        errors="coerce"
    ).fillna(0)

    # IMDb-style Bayesian weighting
    C = vote_average.mean()

    m = vote_count.quantile(
        0.60
    )

    quality = (
        (
            vote_count /
            (vote_count + m)
        ) * vote_average
        +
        (
            m /
            (vote_count + m)
        ) * C
    )

    quality = quality.fillna(
        C
    )

    quality_min = quality.min()
    quality_max = quality.max()

    if quality_max > quality_min:

        quality_scores = (
            quality - quality_min
        ) / (
            quality_max - quality_min
        )

    else:

        quality_scores = pd.Series(
            0.5,
            index=movies.index
        )

else:

    # Fallback: use MovieLens ratings
    movie_stats = train_ratings.groupby(
        "movieId"
    )["rating"].agg(
        ["mean", "count"]
    )

    global_mean = train_ratings[
        "rating"
    ].mean()

    C = global_mean

    m = movie_stats[
        "count"
    ].quantile(0.60)

    movie_quality = (
        (
            movie_stats["count"] /
            (
                movie_stats["count"] + m
            )
        )
        *
        movie_stats["mean"]
        +
        (
            m /
            (
                movie_stats["count"] + m
            )
        )
        * C
    )

    quality_scores = (
        movies["movieId"]
        .map(movie_quality)
        .fillna(C)
    )

    quality_min = quality_scores.min()
    quality_max = quality_scores.max()

    if quality_max > quality_min:

        quality_scores = (
            quality_scores - quality_min
        ) / (
            quality_max - quality_min
        )

    else:

        quality_scores = pd.Series(
            0.5,
            index=movies.index
        )


movies["quality_score"] = (
    quality_scores.values
)

print(
    "Quality scores created."
)


# ============================================================
# CONTENT SCORE
# ============================================================

def get_content_score(
    user_id,
    movie_id
):

    if movie_id not in movie_index:

        return 0.0

    movie_idx = movie_index[
        movie_id
    ]

    # User's liked training movies
    user_train = train_ratings[
        train_ratings["userId"] == user_id
    ]

    liked = user_train[
        user_train["rating"] >= MIN_RATING
    ]

    liked_indices = []

    for liked_movie_id in liked[
        "movieId"
    ]:

        if liked_movie_id in movie_index:

            liked_indices.append(
                movie_index[
                    liked_movie_id
                ]
            )

    if not liked_indices:

        return 0.0

    # Compare target movie against
    # all movies the user liked.
    target_vector = tfidf_matrix[
        movie_idx
    ]

    liked_matrix = tfidf_matrix[
        liked_indices
    ]

    similarities = cosine_similarity(
        target_vector,
        liked_matrix
    ).flatten()

    if len(similarities) == 0:

        return 0.0

    return float(
        similarities.max()
    )


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_collaborative_score(
    score
):

    # Convert 0.5-5 rating into 0-1
    return np.clip(
        (score - 0.5) / 4.5,
        0,
        1
    )


# ============================================================
# CREATE LEARNING EXAMPLES
# ============================================================

print(
    "\nCreating learning examples..."
)

X = []
y = []

# Only use test ratings for learning examples.
# This prevents leakage from using the same
# rating as both training target and feature.

for _, row in test_ratings.iterrows():

    user_id = int(
        row["userId"]
    )

    movie_id = int(
        row["movieId"]
    )

    actual_rating = float(
        row["rating"]
    )

    # User must exist in SVD
    if user_id not in predicted_ratings.index:

        continue

    # Movie must exist in training matrix
    if movie_id not in predicted_ratings.columns:

        continue

    # Collaborative prediction
    collaborative_score = float(
        predicted_ratings.loc[
            user_id,
            movie_id
        ]
    )

    collaborative_norm = (
        normalize_collaborative_score(
            collaborative_score
        )
    )

    # Content score
    content_score = get_content_score(
        user_id,
        movie_id
    )

    # Quality
    movie_idx = movie_index[
        movie_id
    ]

    quality_score = float(
        movies.iloc[
            movie_idx
        ]["quality_score"]
    )

    # Features
    X.append(
        [
            collaborative_norm,
            content_score,
            quality_score
        ]
    )

    # Target
    # Normalize actual rating to 0-1
    y.append(
        (actual_rating - 0.5) / 4.5
    )


X = np.asarray(
    X,
    dtype=float
)

y = np.asarray(
    y,
    dtype=float
)


print(
    "Learning examples:",
    len(X)
)


if len(X) == 0:

    raise ValueError(
        "No learning examples were created. "
        "Check movieId alignment between "
        "ratings.csv and movies_features.csv."
    )


# ============================================================
# TRAIN WEIGHT MODEL
# ============================================================

print(
    "\nTraining weight learner..."
)

weight_model = LinearRegression(
    fit_intercept=True
)

weight_model.fit(
    X,
    y
)


# ============================================================
# LEARNED WEIGHTS
# ============================================================

weights = weight_model.coef_

intercept = (
    weight_model.intercept_
)

print(
    "\n=========================================="
)

print(
    "LEARNED HYBRID WEIGHT MODEL"
)

print(
    "=========================================="
)

print(
    "Collaborative coefficient:",
    weights[0]
)

print(
    "Content coefficient:",
    weights[1]
)

print(
    "Quality coefficient:",
    weights[2]
)

print(
    "Intercept:",
    intercept
)


# ============================================================
# TRAINING PREDICTIONS
# ============================================================

predicted_y = weight_model.predict(
    X
)

predicted_y = np.clip(
    predicted_y,
    0,
    1
)

predicted_ratings_for_eval = (
    predicted_y * 4.5 + 0.5
)

actual_ratings_for_eval = (
    y * 4.5 + 0.5
)


# ============================================================
# EVALUATION
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        actual_ratings_for_eval,
        predicted_ratings_for_eval
    )
)

mae = mean_absolute_error(
    actual_ratings_for_eval,
    predicted_ratings_for_eval
)


print(
    "\nRMSE:",
    f"{rmse:.4f}"
)

print(
    "MAE:",
    f"{mae:.4f}"
)

print(
    "=========================================="
)


# ============================================================
# SAVE WEIGHTS
# ============================================================

weights_df = pd.DataFrame(
    {
        "feature": [
            "collaborative",
            "content",
            "quality"
        ],
        "weight": [
            weights[0],
            weights[1],
            weights[2]
        ]
    }
)

weights_df.to_csv(
    "data/hybrid_weights.csv",
    index=False
)

print(
    "\nSaved weights to:"
)

print(
    "data/hybrid_weights.csv"
)


# ============================================================
# SAVE INTERCEPT
# ============================================================

with open(
    "data/hybrid_intercept.txt",
    "w"
) as f:

    f.write(
        str(intercept)
    )


print(
    "Saved intercept to:"
)

print(
    "data/hybrid_intercept.txt"
)


# ============================================================
# SHOW EXAMPLE
# ============================================================

print(
    "\n=========================================="
)

print(
    "EXAMPLE LEARNED HYBRID FORMULA"
)

print(
    "=========================================="
)

print(
    "Hybrid score = "
    f"({weights[0]:.4f} × Collaborative) + "
    f"({weights[1]:.4f} × Content) + "
    f"({weights[2]:.4f} × Quality) + "
    f"{intercept:.4f}"
)

print(
    "=========================================="
)

print(
    "\nDone."
)