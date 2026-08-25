import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD FEATURES
# ==========================================

movies = pd.read_csv(
    "data/movies_features.csv"
)

print("Movies loaded:", len(movies))


# ==========================================
# CLEAN FEATURES
# ==========================================

movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
)


# ==========================================
# TF-IDF
# ==========================================

print("\nBuilding TF-IDF matrix...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ==========================================
# COSINE SIMILARITY
# ==========================================

print("\nCalculating similarity...")

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

print(
    "Similarity matrix shape:",
    similarity_matrix.shape
)


# ==========================================
# MOVIE LOOKUP
# ==========================================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# ==========================================
# RECOMMEND FUNCTION
# ==========================================

def recommend(
    movie_title,
    n=10
):

    if movie_title not in movie_indices:

        print(
            f"\nMovie not found: {movie_title}"
        )

        return

    idx = movie_indices[
        movie_title
    ]

    scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_idx, score in scores[1:]:

        title = movies.iloc[
            movie_idx
        ]["title"]

        recommendations.append(
            (title, score)
        )

        if len(recommendations) == n:
            break

    print(
        f"\nBecause you liked: {movie_title}"
    )

    print(
        "\nContent-based recommendations:"
    )

    for title, score in recommendations:

        print(
            f"{title} "
            f"(similarity: {score:.3f})"
        )


# ==========================================
# TEST
# ==========================================

recommend(
    "Toy Story (1995)",
    10
)