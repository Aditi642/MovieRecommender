import pandas as pd

links = pd.read_csv(
    "ml-latest-small/links.csv"
)

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"],
    errors="coerce"
)

duplicate_tmdb = links[
    links["tmdbId"].notna()
    & links["tmdbId"].duplicated(keep=False)
]


print("===== DUPLICATE TMDB IDs =====")
print(duplicates.to_string(index=False))

print("\nNumber of duplicate rows:")
print(len(duplicates))