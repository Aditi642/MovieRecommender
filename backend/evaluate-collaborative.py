import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==========================================
# CONFIGURATION
# ==========================================

N_COMPONENTS = 50
TOP_K = 10
MIN_RATING = 4.0
RANDOM_STATE = 42


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
# TRAIN / TEST SPLIT
# ==========================================

print("\nCreating train/test split...")

train_parts = []
test_parts = []

rng = np.random.default_rng(RANDOM_STATE)

for user_id, user_ratings in ratings.groupby("userId"):

    user_ratings = user_ratings.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    # Need at least 5 ratings
    if len(user_ratings) < 5:

        train_parts.append(user_ratings)

        continue

    # Hold out 20%
    test_size = max(
        1,
        int(len(user_ratings) * 0.2)
    )

    test_user = user_ratings.iloc[
        -test_size:
    ]

    train_user = user_ratings.iloc[
        :-test_size
    ]

    train_parts.append(train_user)
    test_parts.append(test_user)


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


# ==========================================
# CREATE TRAINING MATRIX
# ==========================================

train_matrix = train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print(
    "\nTraining matrix:",
    train_matrix.shape
)


# ==========================================
# USER MEANS
# ==========================================

user_means = train_matrix.mean(
    axis=1
)


# ==========================================
# CENTER RATINGS
# ==========================================

centered_matrix = (
    train_matrix
    .sub(user_means, axis=0)
)

centered_matrix = centered_matrix.fillna(0)


# ==========================================
# SVD
# ==========================================

print(
    "\nTraining centered SVD..."
)

svd = TruncatedSVD(
    n_components=N_COMPONENTS,
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


# ==========================================
# PREDICT ALL TRAINING-MATRIX MOVIES
# ==========================================

predicted_centered = (
    user_factors @ movie_factors
)

predicted_ratings = (
    predicted_centered
    + user_means.values.reshape(-1, 1)
)

predicted_ratings = pd.DataFrame(
    predicted_ratings,
    index=train_matrix.index,
    columns=train_matrix.columns
)

predicted_ratings = predicted_ratings.clip(
    lower=0.5,
    upper=5.0
)


# ==========================================
# RMSE / MAE
# ==========================================

actual_values = []
predicted_values = []

for row in test_ratings.itertuples():

    user_id = row.userId
    movie_id = row.movieId
    actual_rating = row.rating

    if user_id not in predicted_ratings.index:
        continue

    if movie_id not in predicted_ratings.columns:
        continue

    prediction = predicted_ratings.loc[
        user_id,
        movie_id
    ]

    actual_values.append(
        actual_rating
    )

    predicted_values.append(
        prediction
    )


print(
    "\nEvaluation ratings:",
    len(actual_values)
)


rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        predicted_values
    )
)

mae = mean_absolute_error(
    actual_values,
    predicted_values
)


# ==========================================
# TOP-K EVALUATION
# ==========================================

print(
    "\nEvaluating Top-K recommendations..."
)


hits = 0
precision_sum = 0
recall_sum = 0
evaluated_users = 0


# Test ratings grouped by user

test_by_user = test_ratings.groupby(
    "userId"
)


for user_id, user_test in test_by_user:

    if user_id not in predicted_ratings.index:
        continue

    # Movies that were actually liked
    relevant_movies = set(
        user_test[
            user_test["rating"] >= MIN_RATING
        ]["movieId"]
    )

    if not relevant_movies:
        continue

    # Get predictions
    scores = predicted_ratings.loc[
        user_id
    ].copy()

    # Remove movies already seen in training
    train_seen = set(
        train_ratings[
            train_ratings["userId"] == user_id
        ]["movieId"]
    )

    scores = scores[
        ~scores.index.isin(train_seen)
    ]

    # Top K
    top_k_movies = scores.sort_values(
        ascending=False
    ).head(TOP_K)

    recommended_movies = set(
        top_k_movies.index
    )

    # Hits
    hits_for_user = (
        recommended_movies
        & relevant_movies
    )

    number_of_hits = len(
        hits_for_user
    )

    # Precision
    precision = (
        number_of_hits / TOP_K
    )

    # Recall
    recall = (
        number_of_hits
        / len(relevant_movies)
    )

    precision_sum += precision
    recall_sum += recall

    if number_of_hits > 0:
        hits += 1

    evaluated_users += 1


# ==========================================
# FINAL RESULTS
# ==========================================

if evaluated_users > 0:

    precision_at_k = (
        precision_sum
        / evaluated_users
    )

    recall_at_k = (
        recall_sum
        / evaluated_users
    )

    hit_rate_at_k = (
        hits
        / evaluated_users
    )

else:

    precision_at_k = 0
    recall_at_k = 0
    hit_rate_at_k = 0


# ==========================================
# DISPLAY
# ==========================================

print(
    "\n=========================================="
)

print(
    "COLLABORATIVE FILTERING EVALUATION"
)

print(
    "=========================================="
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)

print(
    f"Users evaluated: {evaluated_users}"
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
    "=========================================="
)