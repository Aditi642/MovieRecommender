import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

N_COMPONENTS = 50
TOP_K = 10

COLLAB_WEIGHT = 0.50
CONTENT_WEIGHT = 0.35
QUALITY_WEIGHT = 0.15


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

print(
    "Ratings:",
    len(ratings)
)

print(
    "Movies:",
    len(movies)
)


# ============================================================
# CLEAN MOVIE DATA
# ============================================================

# MovieLens ratings only contain 9724 unique movies.
# Keep only movies that can actually be recommended.

rated_movie_ids = set(
    ratings["movieId"]
)

movies = movies[
    movies["movieId"].isin(
        rated_movie_ids
    )
].copy()

movies = movies.reset_index(
    drop=True
)

print(
    "Movies used:",
    len(movies)
)


# ============================================================
# CONTENT-BASED MODEL
# ============================================================

print(
    "\nBuilding TF-IDF..."
)

movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
)


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
# COLLABORATIVE FILTERING
# ============================================================

print(
    "\nBuilding collaborative model..."
)


# User × Movie matrix

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)


print(
    "User-movie matrix:",
    user_movie_matrix.shape
)


# ============================================================
# CENTER RATINGS BY USER
# ============================================================

user_means = (
    user_movie_matrix
    .mean(axis=1)
)


centered_matrix = (
    user_movie_matrix
    .sub(
        user_means,
        axis=0
    )
)


# Missing ratings become zero
# AFTER centering.

centered_matrix = (
    centered_matrix
    .fillna(0)
)


# ============================================================
# TRAIN SVD
# ============================================================

print(
    "\nTraining SVD..."
)


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


# ============================================================
# PREDICT RATINGS
# ============================================================

predicted_centered = (
    user_factors
    @ movie_factors
)


predicted_ratings = (
    predicted_centered
    + user_means.values.reshape(
        -1,
        1
    )
)


predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


# Keep ratings inside MovieLens range

predicted_ratings = (
    predicted_ratings
    .clip(
        lower=0.5,
        upper=5.0
    )
)


print(
    "Collaborative model ready."
)


# ============================================================
# QUALITY MODEL
# ============================================================

print(
    "\nBuilding movie quality scores..."
)


# IMPORTANT:
# We keep quality information separate from
# movies_features.csv.
#
# This prevents pandas from creating:
#
# vote_average_x
# vote_average_y
#
# etc.

quality = master_movies[
    [
        "movieId",
        "vote_average",
        "vote_count",
        "popularity"
    ]
].copy()


# ============================================================
# REMOVE DUPLICATE MOVIE IDs
# ============================================================

quality = quality.drop_duplicates(
    subset="movieId"
)


# ============================================================
# CONVERT QUALITY COLUMNS TO NUMERIC
# ============================================================

quality["vote_average"] = pd.to_numeric(
    quality["vote_average"],
    errors="coerce"
)

quality["vote_count"] = pd.to_numeric(
    quality["vote_count"],
    errors="coerce"
)

quality["popularity"] = pd.to_numeric(
    quality["popularity"],
    errors="coerce"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

vote_average_median = (
    quality["vote_average"]
    .median()
)


quality["vote_average"] = (
    quality["vote_average"]
    .fillna(
        vote_average_median
    )
)


quality["vote_count"] = (
    quality["vote_count"]
    .fillna(0)
)


quality["popularity"] = (
    quality["popularity"]
    .fillna(0)
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            0.5,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# QUALITY FEATURES
# ============================================================

quality["vote_score"] = (
    min_max_normalize(
        quality["vote_average"]
    )
)


quality["popularity_score"] = (
    min_max_normalize(
        quality["popularity"]
    )
)


# Log transform vote count.
#
# This prevents movies with thousands of
# votes from completely dominating the score.

quality["vote_count_log"] = (
    np.log1p(
        quality["vote_count"]
    )
)


quality["vote_count_score"] = (
    min_max_normalize(
        quality["vote_count_log"]
    )
)


# ============================================================
# FINAL QUALITY SCORE
# ============================================================

quality["quality_score"] = (

    0.50
    * quality["vote_score"]

    +

    0.30
    * quality["vote_count_score"]

    +

    0.20
    * quality["popularity_score"]
)


# ============================================================
# QUALITY LOOKUP
# ============================================================

quality_lookup = pd.Series(
    quality["quality_score"].values,
    index=quality["movieId"]
)


print(
    "Quality scores created."
)


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    minimum = scores.min()
    maximum = scores.max()

    if (
        len(scores) == 0
        or maximum == minimum
    ):

        return np.zeros(
            len(scores)
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# HYBRID RECOMMENDER
# ============================================================

def recommend_for_user(
    user_id,
    n=10
):

    print(
        "\n=========================================="
    )

    print(
        f"HYBRID RECOMMENDATIONS FOR USER {user_id}"
    )

    print(
        "=========================================="
    )


    # ========================================================
    # CHECK USER
    # ========================================================

    if user_id not in predicted_ratings.index:

        print(
            f"User {user_id} not found."
        )

        return


    # ========================================================
    # COLLABORATIVE SCORES
    # ========================================================

    collaborative_scores = (
        predicted_ratings
        .loc[user_id]
        .copy()
    )


    # ========================================================
    # MOVIES ALREADY WATCHED
    # ========================================================

    watched_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )


    # ========================================================
    # CONTENT USER PROFILE
    # ========================================================

    liked_ratings = ratings[
        (
            ratings["userId"]
            == user_id
        )
        &
        (
            ratings["rating"]
            >= 4.0
        )
    ]


    liked_indices = []


    for movie_id in liked_ratings[
        "movieId"
    ]:

        if movie_id in movie_index:

            liked_indices.append(
                movie_index[movie_id]
            )


    # ========================================================
    # CONTENT SCORES
    # ========================================================

    if liked_indices:

        # Average the TF-IDF vectors
        # of movies the user liked.

        user_profile = (
            tfidf_matrix[
                liked_indices
            ].mean(
                axis=0
            )
        )


        # Convert numpy matrix to
        # normal ndarray.

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

        # User has no movies rated >= 4.

        content_scores = np.zeros(
            len(movies)
        )


    # ========================================================
    # BUILD CANDIDATES
    # ========================================================

    candidate_rows = []


    for movie_id in movies[
        "movieId"
    ]:

        # Don't recommend watched movies.

        if movie_id in watched_movies:

            continue


        # Collaborative model must know
        # this movie.

        if movie_id not in collaborative_scores.index:

            continue


        # Movie must exist in content index.

        if movie_id not in movie_index:

            continue


        idx = movie_index[
            movie_id
        ]


        collaborative_score = (
            collaborative_scores[
                movie_id
            ]
        )


        content_score = (
            content_scores[
                idx
            ]
        )


        # Get quality score.
        #
        # If missing, use neutral 0.5.

        quality_score = (
            quality_lookup.get(
                movie_id,
                0.5
            )
        )


        candidate_rows.append(
            {
                "movieId":
                    movie_id,

                "collaborative_score":
                    collaborative_score,

                "content_score":
                    content_score,

                "quality_score":
                    quality_score
            }
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    candidates = pd.DataFrame(
        candidate_rows
    )


    if candidates.empty:

        print(
            "No candidate movies found."
        )

        return


    # ========================================================
    # NORMALIZE COLLABORATIVE SCORES
    # ========================================================

    candidates[
        "collaborative_normalized"
    ] = normalize_scores(
        candidates[
            "collaborative_score"
        ]
    )


    # ========================================================
    # NORMALIZE CONTENT SCORES
    # ========================================================

    candidates[
        "content_normalized"
    ] = normalize_scores(
        candidates[
            "content_score"
        ]
    )


    # ========================================================
    # NORMALIZE QUALITY SCORES
    # ========================================================

    candidates[
        "quality_normalized"
    ] = normalize_scores(
        candidates[
            "quality_score"
        ]
    )


    # ========================================================
    # HYBRID SCORE
    # ========================================================

    candidates[
        "hybrid_score"
    ] = (

        COLLAB_WEIGHT
        * candidates[
            "collaborative_normalized"
        ]

        +

        CONTENT_WEIGHT
        * candidates[
            "content_normalized"
        ]

        +

        QUALITY_WEIGHT
        * candidates[
            "quality_normalized"
        ]
    )


    # ========================================================
    # SORT
    # ========================================================

    candidates = candidates.sort_values(
        by="hybrid_score",
        ascending=False
    )


    # ========================================================
    # TOP N
    # ========================================================

    recommendations = (
        candidates
        .head(n)
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    for _, row in recommendations.iterrows():

        movie_id = row[
            "movieId"
        ]


        # Get movie title.

        title_rows = movies[
            movies["movieId"]
            == movie_id
        ]


        if title_rows.empty:

            title = "Unknown Movie"

        else:

            title = title_rows.iloc[0][
                "title"
            ]


        print(
            f"\n{title}"
        )


        print(
            f"  Hybrid score: "
            f"{row['hybrid_score']:.3f}"
        )


        print(
            f"  Collaborative: "
            f"{row['collaborative_score']:.2f}"
        )


        print(
            f"  Content: "
            f"{row['content_score']:.3f}"
        )


        print(
            f"  Quality: "
            f"{row['quality_score']:.3f}"
        )


# ============================================================
# TEST USERS
# ============================================================

test_users = [
    1,
    10,
    50,
    100,
    200
]


for user_id in test_users:

    recommend_for_user(
        user_id=user_id,
        n=TOP_K
    )