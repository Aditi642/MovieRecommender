import pandas as pd

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Basic information
print("\n===== MOVIES =====")
print(movies.head())

print("\n===== RATINGS =====")
print(ratings.head())

# Dataset sizes
print("\n===== DATASET SIZE =====")
print("Number of movies:", len(movies))
print("Number of ratings:", len(ratings))
print("Number of users:", ratings["userId"].nunique())

# Columns
print("\n===== COLUMNS =====")
print("Movies:", movies.columns.tolist())
print("Ratings:", ratings.columns.tolist())

# Missing values
print("\n===== MISSING VALUES =====")
print("Movies:")
print(movies.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())

# Duplicate rows
print("\n===== DUPLICATES =====")
print("Duplicate movies:", movies.duplicated().sum())
print("Duplicate ratings:", ratings.duplicated().sum())

# Rating statistics
print("\n===== RATING STATISTICS =====")
print(ratings["rating"].describe())

# Most rated movies
rating_counts = ratings.groupby("movieId").size().sort_values(ascending=False)

print("\n===== MOST RATED MOVIES =====")
print(rating_counts.head(10))


# ==============================
# USER ACTIVITY
# ==============================

user_rating_counts = ratings.groupby("userId").size()

print("\n===== USER ACTIVITY =====")
print("Average ratings per user:", user_rating_counts.mean())
print("Most active user:", user_rating_counts.idxmax())
print("Most active user's ratings:", user_rating_counts.max())


# ==============================
# MOVIE RATING INFORMATION
# ==============================

movie_rating_stats = ratings.groupby("movieId")["rating"].agg(
    ["count", "mean"]
)

print("\n===== MOVIE RATING INFORMATION =====")
print("Average ratings per movie:", movie_rating_stats["count"].mean())

print("\nHighest rated movies by average rating:")
print(
    movie_rating_stats
    .sort_values("mean", ascending=False)
    .head(10)
)


# ==============================
# RATING DISTRIBUTION
# ==============================

print("\n===== RATING DISTRIBUTION =====")
print(ratings["rating"].value_counts().sort_index())