import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_macro_fnr(model_name_lower):
    """
    Reads the saved confusion matrix CSV for a model and computes the Macro False Negative Rate.
    """
    cm_path = f"results/{model_name_lower}_confusion_matrix.csv"
    if not os.path.exists(cm_path):
        return 0.0
    
    # Read confusion matrix (dropping the index column name)
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.values
    
    fnr_per_class = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp  # Row sum minus True Positives
        
        if (tp + fn) > 0:
            fnr_per_class.append(fn / (tp + fn))
        else:
            fnr_per_class.append(0.0)
            
    return np.mean(fnr_per_class)

def main():
    result_files = [
        "results/decision_tree_results.csv",
        "results/random_forest_results.csv",
        "results/xgboost_results.csv",
        "results/lightgbm_results.csv",
        "results/catboost_results.csv",
        "results/balanced_random_forest_results.csv"
    ]

    # Verify all files exist
    missing_files = [f for f in result_files if not os.path.exists(f)]
    if missing_files:
        print("Missing result files:", missing_files)
        print("Please run all model scripts first.")
        raise SystemExit

    # Load and aggregate data
    results = [pd.read_csv(f) for f in result_files]
    comparison = pd.concat(results, ignore_index=True)
    
    # Shorten names for clean graph labels (matching your image_2b0d9f.png style)
    name_mapping = {
        "Decision Tree": "DT", "Random Forest": "RF", "XGBoost": "XGB",
        "LightGBM": "LGBM", "CatBoost": "CAT", "Balanced Random Forest": "BRF"
    }
    comparison["Model Short"] = comparison["Model"].map(name_mapping)
    
    # Calculate FNR dynamically for each model from its saved confusion matrix
    comparison["Macro FNR"] = comparison["Model"].apply(
        lambda x: calculate_macro_fnr(x.lower().replace(" ", "_"))
    )

    # Save final structured CSV report
    comparison.to_csv("results/flat_model_comparison.csv", index=False)
    print("\nSaved comparison table to results/flat_model_comparison.csv")
    print(comparison[["Model", "Accuracy", "Macro F1", "Macro FNR"]])

    # Create directories for figures
    os.makedirs("results/figures", exist_ok=True)
    
    # Define distinct color palette matching the reference image style
    colors = ['#e65141', '#4196e6', '#34ca7e', '#f39c12', '#9b59b6', '#e91e63']

    # =========================================================================
    # FIGURE 1: Multi-panel Bar Chart (Accuracy, Macro F1, False Negative Rate)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ["Accuracy", "Macro F1", "Macro FNR"]
    titles = ["Accuracy", "Macro F1", "False Negative Rate (↓ better)"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        bars = ax.bar(comparison["Model Short"], comparison[metric], color=colors, edgecolor='grey', alpha=0.9)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Determine clean y-limits dynamically
        y_min = max(0, comparison[metric].min() - 0.04)
        y_max = min(1.0, comparison[metric].max() + 0.02)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='x', rotation=30)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle("Flat Pipeline — Final Performance Visualizations", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/01_final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 2: Accuracy vs. FNR Trade-off Scatter Plot
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    for idx, row in comparison.iterrows():
        plt.scatter(row["Macro FNR"], row["Accuracy"], color=colors[idx], s=180, edgecolors='black', zorder=3)
        plt.text(row["Macro FNR"] + 0.001, row["Accuracy"] + 0.001, row["Model Short"], 
                 fontsize=10, fontweight='bold', color=colors[idx])

    plt.title("Accuracy vs FNR Trade-off\n(ideal = top-left corner)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Negative Rate (missed attacks ↓ better)", fontsize=11)
    plt.ylabel("Overall Accuracy (↑ better)", fontsize=11)
    
    # Pad margins slightly so labels don't get cut off
    plt.xlim(comparison["Macro FNR"].min() - 0.005, comparison["Macro FNR"].max() + 0.01)
    plt.ylim(comparison["Accuracy"].min() - 0.01, comparison["Accuracy"].max() + 0.01)
    
    plt.tight_layout()
    plt.savefig("results/figures/02_accuracy_vs_fnr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✔️ Successfully generated automated charts inside results/figures/")

if __name__ == "__main__":
    main()