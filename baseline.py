import pandas as pd

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Calculate rating statistics for each movie
movie_stats = ratings.groupby("movieId")["rating"].agg(
    rating_count="count",
    average_rating="mean"
).reset_index()

# Merge movie information with rating statistics
movie_stats = movie_stats.merge(
    movies,
    on="movieId"
)

# Minimum number of ratings required
minimum_ratings = 50

popular_movies = movie_stats[
    movie_stats["rating_count"] >= minimum_ratings
].copy()

# Calculate weighted rating
C = popular_movies["average_rating"].mean()
m = minimum_ratings

popular_movies["weighted_rating"] = (
    (popular_movies["rating_count"] /
     (popular_movies["rating_count"] + m))
    * popular_movies["average_rating"]
    +
    (m /
     (popular_movies["rating_count"] + m))
    * C
)

# Sort by weighted rating
popular_movies = popular_movies.sort_values(
    "weighted_rating",
    ascending=False
)

print("\n===== TOP MOVIES =====")

print(
    popular_movies[
        [
            "title",
            "rating_count",
            "average_rating",
            "weighted_rating"
        ]
    ].head(20).to_string(index=False)
)