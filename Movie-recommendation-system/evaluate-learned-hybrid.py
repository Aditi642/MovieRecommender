import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies_features.csv"

WEIGHTS_PATH = "data/hybrid_weights.csv"
INTERCEPT_PATH = "data/hybrid_intercept.txt"

N_COMPONENTS = 50
MAX_FEATURES = 50000

TOP_K = 10
MIN_RATING = 4.0

MAX_USERS = 100

RANDOM_STATE = 42


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

print(
    "Ratings:",
    len(ratings)
)

print(
    "Movies:",
    len(movies)
)


# ============================================================
# CLEAN
# ============================================================

ratings["movieId"] = (
    ratings["movieId"]
    .astype(int)
)

ratings["userId"] = (
    ratings["userId"]
    .astype(int)
)

movies["movieId"] = (
    movies["movieId"]
    .astype(int)
)

movies = movies.drop_duplicates(
    "movieId"
).reset_index(drop=True)


# ============================================================
# CREATE FEATURES IF NEEDED
# ============================================================

if "combined_features" not in movies.columns:

    print(
        "\ncombined_features not found."
    )

    feature_columns = [
        "genres_clean",
        "keywords_clean",
        "cast_clean",
        "director_clean"
    ]

    available = [
        c
        for c in feature_columns
        if c in movies.columns
    ]

    if not available:

        raise ValueError(
            "No content feature columns found."
        )

    movies["combined_features"] = (
        movies[available]
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
# PER-USER TRAIN / TEST SPLIT
# ============================================================

print(
    "\nCreating per-user train/test split..."
)

rng = np.random.RandomState(
    RANDOM_STATE
)

train_parts = []
test_parts = []

for user_id, group in ratings.groupby(
    "userId"
):

    group = group.copy()

    if len(group) < 2:

        train_parts.append(group)

        continue

    test_size = max(
        1,
        int(round(len(group) * 0.20))
    )

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

test_ratings = pd.concat(
    test_parts,
    ignore_index=True
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
# MOVIE INDEX
# ============================================================

movie_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).drop_duplicates()


# ============================================================
# TF-IDF
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
# TRAINING MATRIX
# ============================================================

print(
    "\nBuilding collaborative model..."
)

training_matrix = train_ratings.pivot_table(
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
# SVD
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
# COLLABORATIVE PREDICTIONS
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
# QUALITY SCORES
# ============================================================

print(
    "\nBuilding movie quality scores..."
)

if (
    "vote_average" in movies.columns
    and
    "vote_count" in movies.columns
):

    vote_average = pd.to_numeric(
        movies["vote_average"],
        errors="coerce"
    ).fillna(0)

    vote_count = pd.to_numeric(
        movies["vote_count"],
        errors="coerce"
    ).fillna(0)

    C = vote_average.mean()

    m = vote_count.quantile(
        0.60
    )

    quality = (
        (
            vote_count /
            (vote_count + m)
        )
        * vote_average
        +
        (
            m /
            (vote_count + m)
        )
        * C
    )

    quality = quality.fillna(C)

else:

    stats = train_ratings.groupby(
        "movieId"
    )["rating"].agg(
        ["mean", "count"]
    )

    C = train_ratings[
        "rating"
    ].mean()

    m = stats["count"].quantile(
        0.60
    )

    quality = (
        (
            stats["count"] /
            (stats["count"] + m)
        )
        *
        stats["mean"]
        +
        (
            m /
            (stats["count"] + m)
        )
        * C
    )

    quality = (
        movies["movieId"]
        .map(quality)
        .fillna(C)
    )


q_min = quality.min()
q_max = quality.max()

if q_max > q_min:

    quality = (
        quality - q_min
    ) / (
        q_max - q_min
    )

else:

    quality = pd.Series(
        0.5,
        index=movies.index
    )


movies["quality_score"] = (
    quality.values
)

print(
    "Quality scores created."
)


# ============================================================
# LOAD LEARNED WEIGHTS
# ============================================================

print(
    "\nLoading learned weights..."
)

weights_df = pd.read_csv(
    WEIGHTS_PATH
)

weights = dict(
    zip(
        weights_df["feature"],
        weights_df["weight"]
    )
)

with open(
    INTERCEPT_PATH,
    "r"
) as f:

    intercept = float(
        f.read()
    )


collab_weight = weights[
    "collaborative"
]

content_weight = weights[
    "content"
]

quality_weight = weights[
    "quality"
]


print(
    "Collaborative weight:",
    collab_weight
)

print(
    "Content weight:",
    content_weight
)

print(
    "Quality weight:",
    quality_weight
)

print(
    "Intercept:",
    intercept
)


# ============================================================
# NORMALIZE COLLABORATIVE SCORE
# ============================================================

def normalize_collaborative(
    score
):

    return np.clip(
        (score - 0.5) / 4.5,
        0,
        1
    )


# ============================================================
# CONTENT PROFILE
# ============================================================

def build_user_profile(
    user_id
):

    if user_id not in train_ratings[
        "userId"
    ].values:

        return None

    user_data = train_ratings[
        train_ratings["userId"] == user_id
    ]

    liked = user_data[
        user_data["rating"] >= MIN_RATING
    ]

    indices = []

    for movie_id in liked[
        "movieId"
    ]:

        if movie_id in movie_index:

            indices.append(
                movie_index[movie_id]
            )

    if not indices:

        return None

    profile = tfidf_matrix[
        indices
    ].mean(axis=0)

    return np.asarray(
        profile
    )


# ============================================================
# RECOMMEND FOR USER
# ============================================================

def recommend_for_user(
    user_id,
    n=10
):

    if user_id not in predicted_ratings.index:

        return []

    profile = build_user_profile(
        user_id
    )

    # Movies already watched
    watched = set(
        train_ratings[
            train_ratings["userId"] == user_id
        ]["movieId"]
    )

    candidate_ids = []

    candidate_indices = []

    for movie_id in predicted_ratings.columns:

        if movie_id in watched:
            continue

        if movie_id not in movie_index:
            continue

        candidate_ids.append(
            movie_id
        )

        candidate_indices.append(
            movie_index[movie_id]
        )

    if not candidate_indices:

        return []

    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    collab_scores = []

    for movie_id in candidate_ids:

        score = predicted_ratings.loc[
            user_id,
            movie_id
        ]

        collab_scores.append(
            normalize_collaborative(
                score
            )
        )

    collab_scores = np.asarray(
        collab_scores
    )

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    if profile is not None:

        content_scores = (
            cosine_similarity(
                profile,
                tfidf_matrix[
                    candidate_indices
                ]
            ).flatten()
        )

    else:

        content_scores = np.zeros(
            len(candidate_ids)
        )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality_scores = (
        movies.iloc[
            candidate_indices
        ]["quality_score"]
        .values
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid_scores = (
        collab_weight
        * collab_scores
        +
        content_weight
        * content_scores
        +
        quality_weight
        * quality_scores
        +
        intercept
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    order = np.argsort(
        hybrid_scores
    )[::-1]

    order = order[:n]

    results = []

    for position in order:

        movie_id = candidate_ids[
            position
        ]

        movie_idx = movie_index[
            movie_id
        ]

        title = movies.iloc[
            movie_idx
        ]["title"]

        results.append(
            (
                movie_id,
                title,
                hybrid_scores[position]
            )
        )

    return results


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating hybrid model..."
)


# Users with enough training/test ratings
train_counts = train_ratings.groupby(
    "userId"
).size()

test_counts = test_ratings.groupby(
    "userId"
).size()

eligible_users = [
    user_id
    for user_id in test_counts.index
    if (
        user_id in train_counts.index
        and train_counts[user_id] >= 5
        and test_counts[user_id] >= 1
    )
]

eligible_users = eligible_users[
    :MAX_USERS
]

print(
    "Eligible users:",
    len(eligible_users)
)


hits = 0

precision_values = []
recall_values = []
ndcg_values = []

evaluated_users = 0


# ============================================================
# TOP-K EVALUATION
# ============================================================

for user_id in eligible_users:

    recommendations = (
        recommend_for_user(
            user_id,
            TOP_K
        )
    )

    if not recommendations:

        continue

    recommended_ids = [
        movie_id
        for movie_id, title, score
        in recommendations
    ]

    # Relevant test movies:
    # rating >= 4
    user_test = test_ratings[
        test_ratings["userId"] == user_id
    ]

    relevant_ids = set(
        user_test[
            user_test["rating"] >= MIN_RATING
        ]["movieId"]
    )

    # Only evaluate movies that exist
    # in our model.
    relevant_ids = {
        movie_id
        for movie_id in relevant_ids
        if movie_id in movie_index
    }

    if not relevant_ids:

        continue

    # --------------------------------------------------------
    # Hits
    # --------------------------------------------------------

    hits_for_user = [
        movie_id
        for movie_id in recommended_ids
        if movie_id in relevant_ids
    ]

    hit_count = len(
        hits_for_user
    )

    if hit_count > 0:

        hits += 1

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        hit_count /
        TOP_K
    )

    precision_values.append(
        precision
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        hit_count /
        len(relevant_ids)
    )

    recall_values.append(
        recall
    )

    # --------------------------------------------------------
    # NDCG
    # --------------------------------------------------------

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended_ids,
        start=1
    ):

        if movie_id in relevant_ids:

            dcg += (
                1.0 /
                np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant_ids),
        TOP_K
    )

    idcg = sum(
        1.0 /
        np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg > 0:

        ndcg = dcg / idcg

    else:

        ndcg = 0.0

    ndcg_values.append(
        ndcg
    )

    evaluated_users += 1


# ============================================================
# FINAL RESULTS
# ============================================================

if evaluated_users == 0:

    print(
        "\nNo users could be evaluated."
    )

    raise SystemExit


precision_at_k = np.mean(
    precision_values
)

recall_at_k = np.mean(
    recall_values
)

hit_rate_at_k = (
    hits /
    evaluated_users
)

ndcg_at_k = np.mean(
    ndcg_values
)


print(
    "\n=========================================="
)

print(
    "LEARNED HYBRID RECOMMENDATION EVALUATION"
)

print(
    "=========================================="
)

print(
    "Users evaluated:",
    evaluated_users
)

print(
    f"Precision@{TOP_K}: "
    f"{precision_at_k:.4f}"
)

print(
    f"Recall@{TOP_K}: "
    f"{recall_at_k:.4f}"
)

print(
    f"Hit Rate@{TOP_K}: "
    f"{hit_rate_at_k:.4f}"
)

print(
    f"NDCG@{TOP_K}: "
    f"{ndcg_at_k:.4f}"
)

print(
    "=========================================="
)


# ============================================================
# EXAMPLE RECOMMENDATIONS
# ============================================================

for user_id in [
    1,
    10,
    50,
    100,
    200
]:

    if user_id not in predicted_ratings.index:

        continue

    recommendations = (
        recommend_for_user(
            user_id,
            10
        )
    )

    print(
        f"\n===== USER {user_id} ====="
    )

    for (
        movie_id,
        title,
        score
    ) in recommendations:

        print(
            f"{title} "
            f"(hybrid score: {score:.3f})"
        )


print(
    "\nEvaluation complete."
)