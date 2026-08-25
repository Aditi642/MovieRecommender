import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid movie recommendation system using collaborative filtering, content-based filtering, and movie quality.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/master_movies.csv"
FEATURES_PATH = "data/movies_features.csv"

WEIGHTS_PATH = "data/hybrid_weights.csv"
INTERCEPT_PATH = "data/hybrid_intercept.txt"

N_COMPONENTS = 50
MAX_FEATURES = 50000


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

ratings = pd.read_csv(
    RATINGS_PATH
)

movies = pd.read_csv(
    MOVIES_PATH
)

features = pd.read_csv(
    FEATURES_PATH
)


print(
    "Ratings:",
    len(ratings)
)

print(
    "Movies:",
    len(movies)
)


# ============================================================
# BASIC CLEANING
# ============================================================

ratings["userId"] = pd.to_numeric(
    ratings["userId"],
    errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"],
    errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

movies["movieId"] = pd.to_numeric(
    movies["movieId"],
    errors="coerce"
)

features["movieId"] = pd.to_numeric(
    features["movieId"],
    errors="coerce"
)


ratings = ratings.dropna(
    subset=[
        "userId",
        "movieId",
        "rating"
    ]
)

movies = movies.dropna(
    subset=[
        "movieId"
    ]
)

features = features.dropna(
    subset=[
        "movieId"
    ]
)


ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)
movies["movieId"] = movies["movieId"].astype(int)
features["movieId"] = features["movieId"].astype(int)


# ============================================================
# REMOVE DUPLICATE MOVIES
# ============================================================

movies = movies.drop_duplicates(
    subset=["movieId"]
)

features = features.drop_duplicates(
    subset=["movieId"]
)


print(
    "Movies used:",
    len(movies)
)


# ============================================================
# ALIGN FEATURES WITH MOVIES
# ============================================================

# Keep the same movie order everywhere.

movie_ids = movies["movieId"].values

feature_lookup = features.set_index(
    "movieId"
)


# ============================================================
# BUILD FEATURE TEXT
# ============================================================

print("Building TF-IDF...")


if "combined_features" in features.columns:

    feature_text = (
        features
        .set_index("movieId")
        .reindex(movie_ids)
        ["combined_features"]
        .fillna("")
        .astype(str)
    )

else:

    print(
        "combined_features not found."
    )

    print(
        "Creating combined features from available columns..."
    )

    possible_columns = [
        "genres_clean",
        "keywords_clean",
        "cast_clean",
        "director_clean",
        "overview"
    ]

    available_columns = [
        column
        for column in possible_columns
        if column in features.columns
    ]

    if not available_columns:

        raise ValueError(
            "No usable content features found in movies_features.csv"
        )

    temp = (
        features
        .set_index("movieId")
        .reindex(movie_ids)
    )

    feature_text = (
        temp[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )


feature_text = (
    feature_text
    .fillna("")
    .astype(str)
)


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=MAX_FEATURES,
    ngram_range=(1, 2)
)


tfidf_matrix = tfidf.fit_transform(
    feature_text
)


print(
    "TF-IDF shape:",
    tfidf_matrix.shape
)


# ============================================================
# MOVIE INDEX FOR TF-IDF
# ============================================================

content_movie_index = {
    int(movie_id): index
    for index, movie_id
    in enumerate(movie_ids)
}


# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

print(
    "Building collaborative model..."
)


user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean"
)


print(
    "Training matrix:",
    user_movie_matrix.shape
)


# ============================================================
# USER MEANS
# ============================================================

user_means = (
    user_movie_matrix
    .mean(axis=1)
)


# ============================================================
# CENTER RATINGS
# ============================================================

centered_matrix = (
    user_movie_matrix
    .sub(
        user_means,
        axis=0
    )
)


centered_matrix = (
    centered_matrix
    .fillna(0)
)


# ============================================================
# SVD
# ============================================================

print(
    "Training SVD..."
)


# Avoid requesting more components than mathematically possible.

max_components = min(
    centered_matrix.shape[0] - 1,
    centered_matrix.shape[1] - 1,
    N_COMPONENTS
)


svd = TruncatedSVD(
    n_components=max_components,
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


# ============================================================
# MOVIE IDs USED BY COLLABORATIVE MODEL
# ============================================================

collaborative_movie_ids = (
    user_movie_matrix
    .columns
    .astype(int)
    .tolist()
)


collaborative_movie_index = {
    int(movie_id): index
    for index, movie_id
    in enumerate(
        collaborative_movie_ids
    )
}


user_index = {
    int(user_id): index
    for index, user_id
    in enumerate(
        user_movie_matrix.index
    )
}


# ============================================================
# PREDICT COLLABORATIVE RATINGS
# ============================================================

predicted_centered = (
    user_factors
    @ movie_factors
)


predicted_ratings_matrix = (
    predicted_centered
    + user_means.values.reshape(-1, 1)
)


predicted_ratings_matrix = np.clip(
    predicted_ratings_matrix,
    0.5,
    5.0
)


print(
    "Collaborative model ready."
)


# ============================================================
# MOVIE QUALITY SCORES
# ============================================================

print(
    "Building quality scores..."
)


# Create quality values from master_movies.csv.

quality_data = (
    movies
    .copy()
)


# ------------------------------------------------------------
# Vote average
# ------------------------------------------------------------

if "vote_average" in quality_data.columns:

    vote_average = pd.to_numeric(
        quality_data["vote_average"],
        errors="coerce"
    )

else:

    vote_average = pd.Series(
        0.0,
        index=quality_data.index
    )


# ------------------------------------------------------------
# Vote count
# ------------------------------------------------------------

if "vote_count" in quality_data.columns:

    vote_count = pd.to_numeric(
        quality_data["vote_count"],
        errors="coerce"
    )

else:

    vote_count = pd.Series(
        0.0,
        index=quality_data.index
    )


# ------------------------------------------------------------
# Popularity
# ------------------------------------------------------------

if "popularity" in quality_data.columns:

    popularity = pd.to_numeric(
        quality_data["popularity"],
        errors="coerce"
    )

else:

    popularity = pd.Series(
        0.0,
        index=quality_data.index
    )


vote_average = vote_average.fillna(
    vote_average.median()
)

vote_count = vote_count.fillna(0)

popularity = popularity.fillna(0)


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(series):

    series = series.astype(float)

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


normalized_vote = (
    vote_average / 5.0
)


# Log transforms reduce the effect of huge vote counts.

normalized_votes = min_max_normalize(
    np.log1p(vote_count)
)

normalized_popularity = min_max_normalize(
    np.log1p(popularity)
)


# Quality formula.

quality_scores_all = (
    0.60 * normalized_vote
    + 0.25 * normalized_votes
    + 0.15 * normalized_popularity
)


quality_scores_all = (
    quality_scores_all
    .fillna(0.0)
    .values
)


print(
    "Quality scores created."
)


# ============================================================
# QUALITY LOOKUP
# ============================================================

quality_by_movie_id = {
    int(movie_id): float(score)
    for movie_id, score
    in zip(
        movies["movieId"],
        quality_scores_all
    )
}


# ============================================================
# LOAD LEARNED HYBRID WEIGHTS
# ============================================================

print(
    "Loading learned weights..."
)


weights = pd.read_csv(
    WEIGHTS_PATH
)


print(
    "\n===== HYBRID WEIGHTS ====="
)

print(
    weights
)


print(
    "\n===== COLUMNS ====="
)

print(
    weights.columns.tolist()
)


# Your file contains:
#
# feature,weight
#
# NOT:
#
# feature,coefficient


if "feature" not in weights.columns:

    raise ValueError(
        "hybrid_weights.csv must contain a 'feature' column."
    )


if "weight" not in weights.columns:

    raise ValueError(
        "hybrid_weights.csv must contain a 'weight' column."
    )


weights_dict = dict(
    zip(
        weights["feature"],
        weights["weight"]
    )
)


collaborative_weight = float(
    weights_dict.get(
        "collaborative",
        0.8
    )
)


content_weight = float(
    weights_dict.get(
        "content",
        0.25
    )
)


quality_weight = float(
    weights_dict.get(
        "quality",
        0.43
    )
)


# ============================================================
# INTERCEPT
# ============================================================

with open(
    INTERCEPT_PATH,
    "r"
) as file:

    intercept = float(
        file.read().strip()
    )


print(
    "\nLearned weights loaded:"
)


print(
    "Collaborative:",
    collaborative_weight
)

print(
    "Content:",
    content_weight
)

print(
    "Quality:",
    quality_weight
)

print(
    "Intercept:",
    intercept
)


# ============================================================
# MOVIE TITLE LOOKUP
# ============================================================

movie_title_lookup = {
    int(row["movieId"]): str(row["title"])
    for _, row
    in movies.iterrows()
}


# ============================================================
# CONTENT PROFILE
# ============================================================

def build_user_content_profile(
    user_id
):

    # Ratings made by this user.

    user_ratings = ratings[
        ratings["userId"] == user_id
    ]

    if user_ratings.empty:

        return None


    # Movies rated highly.

    liked = user_ratings[
        user_ratings["rating"] >= 4.0
    ]


    if liked.empty:

        return None


    profile_indices = []

    profile_weights = []


    for _, row in liked.iterrows():

        movie_id = int(
            row["movieId"]
        )

        if movie_id not in content_movie_index:
            continue


        index = content_movie_index[
            movie_id
        ]


        profile_indices.append(
            index
        )


        # Higher ratings receive slightly
        # more importance.

        profile_weights.append(
            float(row["rating"]) - 3.0
        )


    if not profile_indices:

        return None


    movie_vectors = (
        tfidf_matrix[
            profile_indices
        ]
    )


    weights_array = np.asarray(
        profile_weights,
        dtype=float
    )


    # Weighted average profile.

    weighted_profile = (
        movie_vectors.multiply(
            weights_array.reshape(-1, 1)
        )
        .sum(axis=0)
    )


    weighted_profile = np.asarray(
        weighted_profile
    )


    return weighted_profile


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    user_id,
    n=10
):

    user_id = int(user_id)
    n = int(n)


    if n < 1:

        n = 1


    if n > 100:

        n = 100


    # --------------------------------------------------------
    # Check user
    # --------------------------------------------------------

    if user_id not in user_index:

        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found."
        )


    # --------------------------------------------------------
    # User index
    # --------------------------------------------------------

    uidx = user_index[
        user_id
    ]


    # --------------------------------------------------------
    # Seen movies
    # --------------------------------------------------------

    seen_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
        .astype(int)
        .tolist()
    )


    # ========================================================
    # COLLABORATIVE SCORES
    # ========================================================

    collaborative_scores = np.full(
        len(movies),
        user_means.loc[user_id],
        dtype=float
    )


    # Put SVD predictions into the
    # full movie list.

    for movie_position, movie_id in enumerate(
        movies["movieId"].astype(int)
    ):

        if movie_id in collaborative_movie_index:

            model_index = collaborative_movie_index[
                movie_id
            ]

            collaborative_scores[
                movie_position
            ] = predicted_ratings_matrix[
                uidx,
                model_index
            ]


    collaborative_scores = np.clip(
        collaborative_scores,
        0.5,
        5.0
    )


    # ========================================================
    # CONTENT SCORES
    # ========================================================

    user_profile = (
        build_user_content_profile(
            user_id
        )
    )


    if user_profile is None:

        content_scores = np.zeros(
            len(movies),
            dtype=float
        )

    else:

        content_scores = cosine_similarity(
            user_profile,
            tfidf_matrix
        ).flatten()


    # ========================================================
    # QUALITY SCORES
    # ========================================================

    quality_scores = np.array(
        [
            quality_by_movie_id.get(
                int(movie_id),
                0.0
            )
            for movie_id
            in movies["movieId"]
        ],
        dtype=float
    )


    # ========================================================
    # NORMALIZE COLLABORATIVE SCORES
    # ========================================================

    # Collaborative scores are 0.5 - 5.0.
    #
    # Convert to approximately 0 - 1 so that
    # the learned hybrid weights operate on
    # comparable scales.

    collaborative_normalized = (
        collaborative_scores - 0.5
    ) / 4.5


    collaborative_normalized = np.clip(
        collaborative_normalized,
        0.0,
        1.0
    )


    # ========================================================
    # HYBRID SCORE
    # ========================================================

    hybrid_scores = (
        collaborative_weight
        * collaborative_normalized

        + content_weight
        * content_scores

        + quality_weight
        * quality_scores

        + intercept
    )


    # ========================================================
    # REMOVE ALREADY WATCHED MOVIES
    # ========================================================

    candidate_indices = []

    for index, movie_id in enumerate(
        movies["movieId"].astype(int)
    ):

        if movie_id not in seen_movies:

            candidate_indices.append(
                index
            )


    if not candidate_indices:

        return []


    # ========================================================
    # SORT
    # ========================================================

    candidate_indices = sorted(
        candidate_indices,
        key=lambda index:
            hybrid_scores[index],
        reverse=True
    )


    candidate_indices = candidate_indices[
        :n
    ]


    # ========================================================
    # BUILD RESPONSE
    # ========================================================

    results = []


    for index in candidate_indices:

        movie_id = int(
            movies.iloc[index]["movieId"]
        )


        title = movie_title_lookup.get(
            movie_id,
            "Unknown Movie"
        )


        result = {
            "movieId": movie_id,
            "title": title,
            "hybridScore": round(
                float(
                    hybrid_scores[index]
                ),
                4
            ),
            "collaborativeScore": round(
                float(
                    collaborative_scores[index]
                ),
                4
            ),
            "contentScore": round(
                float(
                    content_scores[index]
                ),
                4
            ),
            "qualityScore": round(
                float(
                    quality_scores[index]
                ),
                4
            )
        }


        # Add extra metadata when available.

        movie_row = movies.iloc[
            index
        ]


        if "genres" in movies.columns:

            result["genres"] = str(
                movie_row["genres"]
            )


        if "release_date" in movies.columns:

            release_date = movie_row[
                "release_date"
            ]

            if pd.notna(release_date):

                result["releaseDate"] = str(
                    release_date
                )


        results.append(
            result
        )


    return results


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Movie Recommendation API",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "movies": int(len(movies)),
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "tfidf_features": int(
            tfidf_matrix.shape[1]
        ),
        "svd_components": int(
            movie_factors.shape[0]
        )
    }


# ============================================================
# RECOMMENDATIONS ENDPOINT
# ============================================================

@app.get(
    "/recommendations/{user_id}"
)
def recommendations(
    user_id: int,
    n: int = 10
):

    results = generate_recommendations(
        user_id=user_id,
        n=n
    )


    return {
        "userId": user_id,
        "count": len(results),
        "recommendations": results
    }