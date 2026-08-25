import pandas as pd

weights = pd.read_csv(
    "data/hybrid_weights.csv"
)

print("\n===== HYBRID WEIGHTS =====")
print(weights)

print("\n===== COLUMNS =====")
print(weights.columns.tolist())