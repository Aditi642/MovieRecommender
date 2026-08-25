import pandas as pd


# ==========================================
# 1. LOAD MOVIELENS DATA
# ==========================================

movies = pd.read_csv(
    "data/movies.csv"
)

# ratings = pd.read_csv(
#     "data/ratings.csv"
# )

links = pd.read_csv(
    "ml-latest-small/links.csv"
)


# ==========================================
# 2. LOAD KAGGLE DATA
# ==========================================

metadata = pd.read_csv(
    "archive/movies_metadata.csv",
    low_memory=False
)

credits = pd.read_csv(
    "archive/credits.csv"
)

keywords = pd.read_csv(
    "archive/keywords.csv"
)


print("MovieLens movies:", len(movies))
# print("MovieLens ratings:", len(ratings))
print("MovieLens links:", len(links))

print("Kaggle metadata:", len(metadata))
print("Kaggle credits:", len(credits))
print("Kaggle keywords:", len(keywords))


# ==========================================
# 3. CLEAN TMDB IDs
# ==========================================

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"],
    errors="coerce"
)

metadata["tmdbId"] = pd.to_numeric(
    metadata["id"],
    errors="coerce"
)

credits["tmdbId"] = pd.to_numeric(
    credits["id"],
    errors="coerce"
)

keywords["tmdbId"] = pd.to_numeric(
    keywords["id"],
    errors="coerce"
)


# ==========================================
# 4. REMOVE INVALID TMDB IDs
# ==========================================

links = links.dropna(
    subset=["tmdbId"]
)

metadata = metadata.dropna(
    subset=["tmdbId"]
)

credits = credits.dropna(
    subset=["tmdbId"]
)

keywords = keywords.dropna(
    subset=["tmdbId"]
)


# Convert to integer

links["tmdbId"] = links["tmdbId"].astype(int)
metadata["tmdbId"] = metadata["tmdbId"].astype(int)
credits["tmdbId"] = credits["tmdbId"].astype(int)
keywords["tmdbId"] = keywords["tmdbId"].astype(int)


# ==========================================
# 5. CHECK DUPLICATES
# ==========================================

print("\n===== DUPLICATE CHECK =====")

print(
    "Duplicate MovieLens links:",
    links["tmdbId"].duplicated().sum()
)

print(
    "Duplicate metadata:",
    metadata["tmdbId"].duplicated().sum()
)

print(
    "Duplicate credits:",
    credits["tmdbId"].duplicated().sum()
)

print(
    "Duplicate keywords:",
    keywords["tmdbId"].duplicated().sum()
)


# ==========================================
# 6. KEEP ONLY REQUIRED METADATA
# ==========================================

metadata = metadata[
    [
        "tmdbId",
        "overview",
        "release_date",
        "popularity",
        "vote_average",
        "vote_count"
    ]
]


# ==========================================
# 7. REMOVE DUPLICATES
# ==========================================

metadata = metadata.drop_duplicates(
    subset=["tmdbId"]
)

credits = credits.drop_duplicates(
    subset=["tmdbId"]
)

keywords = keywords.drop_duplicates(
    subset=["tmdbId"]
)


# ==========================================
# 8. JOIN MOVIELENS → TMDB METADATA
# ==========================================

master_movies = movies.merge(
    links[
        ["movieId", "tmdbId"]
    ],
    on="movieId",
    how="left"
)

print("\n===== AFTER MOVIELENS + LINKS =====")
print("Rows:", len(master_movies))
print(
    "Movies with TMDB ID:",
    master_movies["tmdbId"].notna().sum()
)


# ==========================================
# 9. JOIN MOVIE METADATA
# ==========================================

master_movies = master_movies.merge(
    metadata,
    on="tmdbId",
    how="left"
)

print("\n===== AFTER METADATA JOIN =====")
print("Rows:", len(master_movies))
print(
    "Movies with overview:",
    master_movies["overview"].notna().sum()
)


# ==========================================
# 10. JOIN CREDITS
# ==========================================

master_movies = master_movies.merge(
    credits[
        ["tmdbId", "cast", "crew"]
    ],
    on="tmdbId",
    how="left"
)


# ==========================================
# 11. JOIN KEYWORDS
# ==========================================

master_movies = master_movies.merge(
    keywords[
        ["tmdbId", "keywords"]
    ],
    on="tmdbId",
    how="left"
)


# ==========================================
# 12. FINAL CHECK
# ==========================================

print("\n===== FINAL DATASET =====")

print("Shape:", master_movies.shape)

print("\nColumns:")
print(master_movies.columns.tolist())

print("\nMissing values:")
print(
    master_movies.isnull().sum()
)


# ==========================================
# 13. SAVE
# ==========================================

master_movies.to_csv(
    "data/master_movies.csv",
    index=False
)

print(
    "\nSaved to: data/master_movies.csv"
)