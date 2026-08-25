import pandas as pd

# MovieLens links
movielens_links = pd.read_csv(
    "ml-latest-small/links.csv"
)

# Kaggle links
kaggle_links = pd.read_csv(
    "archive/links.csv"
)

print("===== MOVIELENS LINKS =====")
print("Shape:", movielens_links.shape)
print("Columns:", movielens_links.columns.tolist())
print(movielens_links.head())

print("\n===== KAGGLE LINKS =====")
print("Shape:", kaggle_links.shape)
print("Columns:", kaggle_links.columns.tolist())
print(kaggle_links.head())