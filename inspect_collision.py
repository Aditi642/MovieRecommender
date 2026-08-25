import pandas as pd

movies = pd.read_csv("data/movies.csv")
links = pd.read_csv("ml-latest-small/links.csv")

collision_ids = [6003, 144606]

result = movies[
    movies["movieId"].isin(collision_ids)
]

result = result.merge(
    links,
    on="movieId",
    how="left"
)

print("===== MOVIELENS MOVIES =====")
print(result.to_string(index=False))