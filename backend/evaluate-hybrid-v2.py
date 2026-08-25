import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 50
TOP_K = 10

MIN_RATING = 4.0

TEST_RATIO = 0.20

RANDOM_STATE = 42

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
# CLEAN MOVIES
# ============================================================

movies = movies.drop_duplicates(
    subset="movieId"
).copy()

movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
)


valid_movie_ids = set(
    movies["movieId"]
)

ratings = ratings[
    ratings["movieId"].isin(valid_movie_ids)
].copy()


print(
    "Movies used:",
    len(movies)
)


# ============================================================
# PROPER PER-USER TRAIN / TEST SPLIT
# ============================================================

print("\nCreating per-user train/test split...")

rng = np.random.RandomState(
    RANDOM_STATE
)

train_parts = []
test_parts = []

for user_id, user_ratings in ratings.groupby(
    "userId"
):

    user_ratings = user_ratings.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    n_test = max(
        1,
        int(
            len(user_ratings)
            * TEST_RATIO
        )
    )

    # Make sure at least one rating
    # remains for training.
    if n_test >= len(user_ratings):
        n_test = len(user_ratings) - 1

    test_part = user_ratings.iloc[
        :n_test
    ]

    train_part = user_ratings.iloc[
        n_test:
    ]

    train_parts.append(
        train_part
    )

    test_parts.append(
        test_part
    )


train_ratings = pd.concat(
    train_parts
).reset_index(drop=True)

test_ratings = pd.concat(
    test_parts
).reset_index(drop=True)


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

print(
    "\nBuilding collaborative model..."
)


user_movie_matrix = train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)


print(
    "Training matrix:",
    user_movie_matrix.shape
)


# User mean

user_means = (
    user_movie_matrix.mean(
        axis=1
    )
)


# Center ratings

centered_matrix = (
    user_movie_matrix
    .sub(
        user_means,
        axis=0
    )
    .fillna(0)
)


# ============================================================
# SVD
# ============================================================

print("\nTraining SVD...")

# SVD components cannot exceed matrix dimensions.

max_components = min(
    centered_matrix.shape
) - 1

n_components = min(
    N_COMPONENTS,
    max_components
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
# PREDICT RATINGS
# ============================================================

predicted_centered = (
    user_factors
    @
    movie_factors
)


predicted_ratings = (
    predicted_centered
    +
    user_means.values.reshape(
        -1,
        1
    )
)


predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


predicted_ratings = (
    predicted_ratings
    .clip(
        lower=0.5,
        upper=5.0
    )
)


# ============================================================
# QUALITY MODEL
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


quality_data[
    "vote_average"
] = pd.to_numeric(
    quality_data[
        "vote_average"
    ],
    errors="coerce"
).fillna(0)


quality_data[
    "vote_count"
] = pd.to_numeric(
    quality_data[
        "vote_count"
    ],
    errors="coerce"
).fillna(0)


# Normalize vote average

quality_data[
    "rating_norm"
] = (
    quality_data[
        "vote_average"
    ]
    /
    10
)


# Log-transform vote count

quality_data[
    "popularity_norm"
] = np.log1p(
    quality_data[
        "vote_count"
    ]
)


max_popularity = (
    quality_data[
        "popularity_norm"
    ].max()
)


if max_popularity > 0:

    quality_data[
        "popularity_norm"
    ] /= max_popularity


# Final quality score

quality_data[
    "quality"
] = (
    0.7
    *
    quality_data[
        "rating_norm"
    ]
    +
    0.3
    *
    quality_data[
        "popularity_norm"
    ]
)


quality_scores = pd.Series(
    quality_data["quality"].values,
    index=quality_data["movieId"]
)


print(
    "Quality scores created."
)


# ============================================================
# NORMALIZE ARRAY
# ============================================================

def normalize_scores(
    scores
):

    scores = np.asarray(
        scores,
        dtype=float
    )

    minimum = np.min(
        scores
    )

    maximum = np.max(
        scores
    )

    if maximum == minimum:

        return np.zeros_like(
            scores
        )

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# GET HYBRID RECOMMENDATIONS
# ============================================================

def get_hybrid_recommendations(
    user_id,
    top_k=10
):

    if user_id not in predicted_ratings.index:

        return []


    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    collab_scores = (
        predicted_ratings
        .loc[user_id]
    )


    # Convert to dictionary

    collab_dict = (
        collab_scores
        .to_dict()
    )


    # Normalize collaborative scores

    collab_values = np.array(
        list(
            collab_dict.values()
        )
    )


    collab_normalized = (
        normalize_scores(
            collab_values
        )
    )


    collab_dict = dict(
        zip(
            collab_dict.keys(),
            collab_normalized
        )
    )


    # --------------------------------------------------------
    # Movies already seen
    # --------------------------------------------------------

    seen_movies = set(
        train_ratings[
            train_ratings["userId"]
            ==
            user_id
        ]["movieId"]
    )


    # --------------------------------------------------------
    # User's liked movies
    # --------------------------------------------------------

    liked_movies = train_ratings[
        (
            train_ratings["userId"]
            ==
            user_id
        )
        &
        (
            train_ratings["rating"]
            >=
            MIN_RATING
        )
    ]


    profile_indices = []


    for movie_id in (
        liked_movies["movieId"]
    ):

        if movie_id in movie_index:

            profile_indices.append(
                movie_index[movie_id]
            )


    # --------------------------------------------------------
    # Content profile
    # --------------------------------------------------------

    if profile_indices:

        user_profile = (
            tfidf_matrix[
                profile_indices
            ].mean(
                axis=0
            )
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


    content_normalized = (
        normalize_scores(
            content_scores
        )
    )


    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidates = []


    for idx, movie_id in enumerate(
        movies["movieId"]
    ):

        # Don't recommend seen movies

        if movie_id in seen_movies:

            continue


        collaborative = (
            collab_dict.get(
                movie_id,
                0
            )
        )


        content = (
            content_normalized[
                idx
            ]
        )


        quality = (
            quality_scores.get(
                movie_id,
                0
            )
        )


        # Hybrid score

        hybrid_score = (
            COLLAB_WEIGHT
            *
            collaborative
            +
            CONTENT_WEIGHT
            *
            content
            +
            QUALITY_WEIGHT
            *
            quality
        )


        candidates.append(
            (
                movie_id,
                hybrid_score,
                collaborative,
                content,
                quality
            )
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return candidates[
        :top_k
    ]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    recommended,
    relevant
):

    recommended_ids = {
        item[0]
        for item in recommended
    }

    hits = len(
        recommended_ids
        &
        set(relevant)
    )

    return hits / TOP_K


def recall_at_k(
    recommended,
    relevant
):

    if len(relevant) == 0:

        return 0.0


    recommended_ids = {
        item[0]
        for item in recommended
    }


    hits = len(
        recommended_ids
        &
        set(relevant)
    )


    return hits / len(
        relevant
    )


def hit_rate_at_k(
    recommended,
    relevant
):

    recommended_ids = {
        item[0]
        for item in recommended
    }


    return int(
        len(
            recommended_ids
            &
            set(relevant)
        )
        > 0
    )


def ndcg_at_k(
    recommended,
    relevant
):

    relevant = set(
        relevant
    )


    if not relevant:

        return 0.0


    dcg = 0.0


    for position, item in enumerate(
        recommended,
        start=1
    ):

        movie_id = item[0]


        if movie_id in relevant:

            dcg += (
                1
                /
                np.log2(
                    position + 1
                )
            )


    ideal_count = min(
        len(relevant),
        TOP_K
    )


    idcg = sum(
        1
        /
        np.log2(
            position + 1
        )
        for position
        in range(
            1,
            ideal_count + 1
        )
    )


    if idcg == 0:

        return 0.0


    return dcg / idcg


# ============================================================
# EVALUATE
# ============================================================

print(
    "\nEvaluating hybrid model..."
)


eligible_users = []


for user_id, user_test in (
    test_ratings.groupby("userId")
):

    # Need at least one relevant
    # movie in the test set.

    relevant = user_test[
        user_test["rating"]
        >=
        MIN_RATING
    ]["movieId"].tolist()


    if len(relevant) > 0:

        eligible_users.append(
            user_id
        )


eligible_users = eligible_users[
    :MAX_USERS
]


print(
    "Eligible users:",
    len(eligible_users)
)


precision_scores = []
recall_scores = []
hit_scores = []
ndcg_scores = []


# ============================================================
# EVALUATION LOOP
# ============================================================

for counter, user_id in enumerate(
    eligible_users,
    start=1
):

    user_test = test_ratings[
        test_ratings["userId"]
        ==
        user_id
    ]


    relevant_movies = user_test[
        user_test["rating"]
        >=
        MIN_RATING
    ]["movieId"].tolist()


    recommendations = (
        get_hybrid_recommendations(
            user_id,
            TOP_K
        )
    )


    if not recommendations:

        continue


    precision_scores.append(
        precision_at_k(
            recommendations,
            relevant_movies
        )
    )


    recall_scores.append(
        recall_at_k(
            recommendations,
            relevant_movies
        )
    )


    hit_scores.append(
        hit_rate_at_k(
            recommendations,
            relevant_movies
        )
    )


    ndcg_scores.append(
        ndcg_at_k(
            recommendations,
            relevant_movies
        )
    )


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


evaluated = len(
    precision_scores
)


print(
    "Users evaluated:",
    evaluated
)


if evaluated > 0:

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


print(
    "\n=========================================="
)

print(
    "Evaluation complete."
)

print(
    "=========================================="
)