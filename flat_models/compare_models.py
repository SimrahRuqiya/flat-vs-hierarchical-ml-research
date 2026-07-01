import os
import pandas as pd


result_files = [
    "results/decision_tree_results.csv",
    "results/random_forest_results.csv",
    "results/xgboost_results.csv",
    "results/lightgbm_results.csv"
]

missing_files = []

for file in result_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print("The following result files are missing:")
    for file in missing_files:
        print(file)

    print("\nRun the corresponding model scripts first.")
    raise SystemExit

results = []

for file in result_files:
    results.append(pd.read_csv(file))

for file in result_files:
    df = pd.read_csv(file)
    print("\n==============================")
    print(file)
    print(df.head())
    print(df.columns)

comparison = pd.concat(results, ignore_index=True)

comparison = comparison[
    [
        "Model",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Training Time",
        "Prediction Time"
    ]
]

comparison = comparison.sort_values(
    by="Macro F1",
    ascending=False
)

print("\nFlat Model Comparison")
print(comparison)

comparison.to_csv(
    "results/flat_model_comparison.csv",
    index=False
)

print("\nSaved comparison table to results/flat_model_comparison.csv")