import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD DATA
# ==============================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")


# ==============================
# CREATE USER-MOVIE MATRIX
# ==============================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

matrix_filled = user_movie_matrix.fillna(0)


# ==============================
# CALCULATE MOVIE SIMILARITY
# ==============================

movie_similarity = cosine_similarity(matrix_filled.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)


# ==============================
# MOVIE ID → TITLE
# ==============================

movie_id_to_title = dict(
    zip(movies["movieId"], movies["title"])
)

title_to_movie_id = dict(
    zip(movies["title"], movies["movieId"])
)


# ==============================
# RECOMMENDATION FUNCTION
# ==============================

def recommend_movies(movie_title, number_of_recommendations=10):

    if movie_title not in title_to_movie_id:
        print("Movie not found.")
        return

    movie_id = title_to_movie_id[movie_title]

    similar_movies = movie_similarity_df[movie_id].sort_values(
        ascending=False
    )

    # Remove the movie itself
    similar_movies = similar_movies.drop(movie_id)

    print(f"\nBecause you liked: {movie_title}\n")
    print("Recommended movies:\n")

    for similar_movie_id, similarity_score in similar_movies.head(
        number_of_recommendations
    ).items():

        title = movie_id_to_title[similar_movie_id]

        print(
            f"{title} "
            f"(similarity: {similarity_score:.3f})"
        )


# ==============================
# TEST
# ==============================

recommend_movies("Toy Story (1995)")