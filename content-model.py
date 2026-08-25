import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD MOVIES
# ==============================

movies = pd.read_csv("data/movies.csv")


# ==============================
# CLEAN GENRES
# ==============================

movies["genres"] = movies["genres"].fillna("")

# Convert "|" into spaces
movies["genres"] = movies["genres"].str.replace(
    "|",
    " ",
    regex=False
)


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer()

genre_matrix = vectorizer.fit_transform(
    movies["genres"]
)


# ==============================
# CONTENT SIMILARITY
# ==============================

content_similarity = cosine_similarity(
    genre_matrix
)


# ==============================
# MOVIE LOOKUP
# ==============================

title_to_index = pd.Series(
    movies.index,
    index=movies["title"]
).to_dict()


# ==============================
# RECOMMENDATION FUNCTION
# ==============================

def recommend_movies(movie_title, number_of_recommendations=10):

    if movie_title not in title_to_index:
        print("Movie not found.")
        return

    movie_index = title_to_index[movie_title]

    similarity_scores = list(
        enumerate(content_similarity[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself
    similarity_scores = similarity_scores[1:]

    print(f"\nBecause you liked: {movie_title}\n")
    print("Content-based recommendations:\n")

    for index, score in similarity_scores[
        :number_of_recommendations
    ]:

        print(
            f"{movies.iloc[index]['title']} "
            f"(similarity: {score:.3f})"
        )


# ==============================
# TEST
# ==============================

recommend_movies("Toy Story (1995)")