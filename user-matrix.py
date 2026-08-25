import pandas as pd

# Load ratings
ratings = pd.read_csv("data/ratings.csv")

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("\n===== USER-MOVIE MATRIX =====")
print(user_movie_matrix.head())

print("\nMatrix shape:")
print(user_movie_matrix.shape)

print("\nNumber of missing values:")
print(user_movie_matrix.isna().sum().sum())