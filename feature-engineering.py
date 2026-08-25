import pandas as pd
import ast


# ==========================================
# LOAD MASTER DATA
# ==========================================

movies = pd.read_csv(
    "data/master_movies.csv"
)

print("Movies loaded:", len(movies))


# ==========================================
# PARSE JSON-LIKE COLUMNS
# ==========================================

def parse_list(value):

    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


# ==========================================
# EXTRACT KEYWORDS
# ==========================================

def get_keywords(value):

    data = parse_list(value)

    return [
        item.get("name", "").replace(" ", "_")
        for item in data
        if isinstance(item, dict)
    ]


# ==========================================
# EXTRACT CAST
# ==========================================

def get_cast(value, limit=5):

    data = parse_list(value)

    return [
        item.get("name", "").replace(" ", "_")
        for item in data[:limit]
        if isinstance(item, dict)
    ]


# ==========================================
# EXTRACT DIRECTOR
# ==========================================

def get_director(value):

    data = parse_list(value)

    for item in data:

        if (
            isinstance(item, dict)
            and item.get("job") == "Director"
        ):

            return item.get(
                "name",
                ""
            ).replace(" ", "_")

    return ""


# ==========================================
# APPLY FEATURE EXTRACTION
# ==========================================

print("\nExtracting keywords...")

movies["keywords_clean"] = movies[
    "keywords"
].apply(get_keywords)


print("Extracting cast...")

movies["cast_clean"] = movies[
    "cast"
].apply(get_cast)


print("Extracting directors...")

movies["director_clean"] = movies[
    "crew"
].apply(get_director)


# ==========================================
# CLEAN GENRES
# ==========================================

movies["genres_clean"] = (
    movies["genres"]
    .fillna("")
    .str.replace(
        "|",
        " ",
        regex=False
    )
    .str.replace(
        " ",
        "_",
        regex=False
    )
)


# ==========================================
# CLEAN OVERVIEW
# ==========================================

movies["overview_clean"] = (
    movies["overview"]
    .fillna("")
)


# ==========================================
# COMBINE FEATURES
# ==========================================

movies["combined_features"] = (
    # Genres: strong signal
    movies["genres_clean"] + " "
    + movies["genres_clean"] + " "

    # Keywords: strong signal
    + movies["keywords_clean"].apply(
        lambda x: " ".join(x)
    ) + " "
    + movies["keywords_clean"].apply(
        lambda x: " ".join(x)
    ) + " "

    # Director: strong signal
    + movies["director_clean"] + " "
    + movies["director_clean"] + " "

    # Cast: medium signal
    + movies["cast_clean"].apply(
        lambda x: " ".join(x)
    ) + " "

    # Overview: normal signal
    + movies["overview_clean"]
)
# ==========================================
# PREVIEW
# ==========================================

print("\n===== FEATURE PREVIEW =====")

print(
    movies[
        [
            "title",
            "genres_clean",
            "keywords_clean",
            "cast_clean",
            "director_clean"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# SAVE
# ==========================================

movies.to_csv(
    "data/movies_features.csv",
    index=False
)

print(
    "\nSaved to: data/movies_features.csv"
)