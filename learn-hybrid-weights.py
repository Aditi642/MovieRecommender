import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==========================================
# LOAD DATA
# ==========================================

print("Loading data...")

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/master_movies.csv")

print("Ratings:", len(ratings))
print("Movies:", len(movies))


# ==========================================
# CREATE PER-USER TRAIN/TEST SPLIT
# ==========================================

print("\nCreating per-user train/test split...")

train_parts = []
test_parts = []

for user_id, user_ratings in ratings.groupby("userId"):

    if len(user_ratings) < 5:
        train_parts.append(user_ratings)
        continue

    train, test = train_test_split(
        user_ratings,
        test_size=0.20,
        random_state=42
    )

    train_parts.append(train)
    test_parts.append(test)


train_ratings = pd.concat(train_parts)
test_ratings = pd.concat(test_parts)

print("Training ratings:", len(train_ratings))
print("Test ratings:", len(test_ratings))


# ==========================================
# CREATE TRAINING USER-MOVIE MATRIX
# ==========================================

print("\nCreating training matrix...")

train_matrix = train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("Training matrix:", train_matrix.shape)


# ==========================================
# SIMPLE SVD
# ==========================================

from sklearn.decomposition import TruncatedSVD

print("\nTraining SVD...")

user_means = train_matrix.mean(axis=1)

centered = train_matrix.sub(
    user_means,
    axis=0
).fillna(0)

svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_factors = svd.fit_transform(centered)

movie_factors = svd.components_

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

print("SVD ready.")


# ==========================================
# CREATE TRAINING EXAMPLES
# ==========================================

print("\nCreating learning examples...")

examples = []

for _, row in test_ratings.iterrows():

    user_id = row["userId"]
    movie_id = row["movieId"]
    actual_rating = row["rating"]

    # User must exist in training
    if user_id not in predicted_ratings.index:
        continue

    # Movie must exist in training matrix
    if movie_id not in predicted_ratings.columns:
        continue

    collaborative_score = predicted_ratings.loc[
        user_id,
        movie_id
    ]

    examples.append({
        "userId": user_id,
        "movieId": movie_id,
        "actual_rating": actual_rating,
        "collaborative": collaborative_score
    })


examples = pd.DataFrame(examples)

print(
    "Learning examples:",
    len(examples)
)


# ==========================================
# TRAIN LEARNER
# ==========================================

print("\nTraining weight learner...")

X = examples[
    ["collaborative"]
]

y = examples[
    "actual_rating"
]

model = Ridge(
    alpha=1.0
)

model.fit(
    X,
    y
)


# ==========================================
# RESULTS
# ==========================================

predictions = model.predict(X)

rmse = np.sqrt(
    mean_squared_error(
        y,
        predictions
    )
)

mae = mean_absolute_error(
    y,
    predictions
)


print("\n==========================================")
print("LEARNED WEIGHT MODEL")
print("==========================================")

print(
    "Collaborative coefficient:",
    model.coef_[0]
)

print(
    "Intercept:",
    model.intercept_
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)

print("==========================================")