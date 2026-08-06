from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "saved_models"
SHAP_RESULTS_DIR = ROOT_DIR / "results" / "shap" / "no_smote"

SHAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = MODEL_DIR / "random_forest_no_smote.joblib"

LABELS = [0, 1, 2, 3, 4]

LABEL_NAMES = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Spoofing"
]

# Use a sample instead of all 265,529 test rows.
# Start with 1,000. Increase later if your computer handles it.
SAMPLE_SIZE = 1000
RANDOM_STATE = 42
MAX_FEATURES_TO_DISPLAY = 15


# ---------------------------------------------------------
# Verify required files
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found:\n{MODEL_PATH}\n\n"
        "Run flat_models/random_forest.py first."
    )

x_test_path = DATA_DIR / "X_test.csv"
y_test_path = DATA_DIR / "y_test.csv"

if not x_test_path.exists() or not y_test_path.exists():
    raise FileNotFoundError(
        "X_test.csv or y_test.csv could not be found in the data folder."
    )


# ---------------------------------------------------------
# Load model and original test data
# ---------------------------------------------------------

print("Loading trained Random Forest model...")
rf_model = joblib.load(MODEL_PATH)

print("Loading original test data...")
X_test = pd.read_csv(x_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

print("Full X_test shape:", X_test.shape)
print("Full y_test shape:", y_test.shape)

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test do not contain the same number of rows."
    )


# ---------------------------------------------------------
# Sample real test rows
# ---------------------------------------------------------

sample_size = min(SAMPLE_SIZE, len(X_test))

X_shap = X_test.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

# Keep y aligned with the sampled X rows
y_shap = y_test.loc[X_shap.index]

X_shap = X_shap.reset_index(drop=True)
y_shap = y_shap.reset_index(drop=True)

print("\nSHAP sample shape:", X_shap.shape)
print("\nClass distribution in the SHAP sample:")
print(
    y_shap.value_counts()
    .sort_index()
    .rename(index=dict(zip(LABELS, LABEL_NAMES)))
)


# ---------------------------------------------------------
# Generate SHAP values
# ---------------------------------------------------------

print("\nCreating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf_model)

print("Calculating SHAP values...")
raw_shap_values = explainer.shap_values(
    X_shap,
    check_additivity=False
)


# ---------------------------------------------------------
# Normalize SHAP output format
#
# Depending on the SHAP version, multiclass output may be:
#
# 1. A list:
#    one array per class, each shaped samples × features
#
# 2. One ndarray shaped:
#    samples × features × classes
# ---------------------------------------------------------

if isinstance(raw_shap_values, list):
    class_shap_values = raw_shap_values

elif isinstance(raw_shap_values, np.ndarray):
    if raw_shap_values.ndim == 3:
        class_shap_values = [
            raw_shap_values[:, :, class_index]
            for class_index in range(raw_shap_values.shape[2])
        ]

    elif raw_shap_values.ndim == 2:
        class_shap_values = [raw_shap_values]

    else:
        raise ValueError(
            f"Unexpected SHAP array shape: {raw_shap_values.shape}"
        )

else:
    raise TypeError(
        f"Unexpected SHAP output type: {type(raw_shap_values)}"
    )

print(
    "Number of class-specific SHAP arrays:",
    len(class_shap_values)
)

for class_index, values in enumerate(class_shap_values):
    print(
        f"Class {class_index} SHAP shape:",
        values.shape
    )


# ---------------------------------------------------------
# Global feature importance
#
# Average absolute SHAP values across:
# - all sampled rows
# - all classes
# ---------------------------------------------------------

stacked_values = np.stack(
    class_shap_values,
    axis=2
)

mean_absolute_shap = np.abs(stacked_values).mean(
    axis=(0, 2)
)

global_importance = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean Absolute SHAP": mean_absolute_shap
}).sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

global_importance_path = (
    SHAP_RESULTS_DIR
    / "random_forest_no_smote_global_shap_importance.csv"
)

global_importance.to_csv(
    global_importance_path,
    index=False
)

print("\nTop 15 globally important features:")
print(global_importance.head(15).to_string(index=False))


# ---------------------------------------------------------
# Global feature-importance bar chart
# ---------------------------------------------------------

top_global = global_importance.head(
    MAX_FEATURES_TO_DISPLAY
).sort_values(
    by="Mean Absolute SHAP",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_global["Feature"],
    top_global["Mean Absolute SHAP"]
)

plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Feature")
plt.title(
    "Random Forest Global SHAP Importance — No SMOTE"
)

plt.tight_layout()

global_bar_path = (
    SHAP_RESULTS_DIR
    / "random_forest_no_smote_global_shap_bar.png"
)

plt.savefig(
    global_bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Class-specific SHAP plots and rankings
# ---------------------------------------------------------

number_of_classes = min(
    len(class_shap_values),
    len(LABEL_NAMES)
)

for class_index in range(number_of_classes):
    class_name = LABEL_NAMES[class_index]
    safe_class_name = class_name.lower().replace(" ", "_")

    class_values = class_shap_values[class_index]

    # Save class-specific numerical ranking
    class_importance = pd.DataFrame({
        "Feature": X_shap.columns,
        "Mean Absolute SHAP": np.abs(class_values).mean(axis=0)
    }).sort_values(
        by="Mean Absolute SHAP",
        ascending=False
    )

    class_importance.to_csv(
        SHAP_RESULTS_DIR
        / (
            f"random_forest_no_smote_"
            f"{safe_class_name}_shap_importance.csv"
        ),
        index=False
    )

    # SHAP beeswarm-style summary plot
    plt.figure()

    shap.summary_plot(
        class_values,
        X_shap,
        feature_names=X_shap.columns,
        max_display=MAX_FEATURES_TO_DISPLAY,
        show=False
    )

    plt.title(
        f"Random Forest SHAP Summary — {class_name} — No SMOTE"
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_RESULTS_DIR
        / (
            f"random_forest_no_smote_"
            f"{safe_class_name}_shap_summary.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Class-specific bar plot
    plt.figure()

    shap.summary_plot(
        class_values,
        X_shap,
        feature_names=X_shap.columns,
        plot_type="bar",
        max_display=MAX_FEATURES_TO_DISPLAY,
        show=False
    )

    plt.title(
        f"Random Forest SHAP Importance — {class_name} — No SMOTE"
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_RESULTS_DIR
        / (
            f"random_forest_no_smote_"
            f"{safe_class_name}_shap_bar.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------
# Save sampled test rows for reproducibility
# ---------------------------------------------------------

sample_output = X_shap.copy()
sample_output["true_label"] = y_shap
sample_output["true_class_name"] = y_shap.map(
    dict(zip(LABELS, LABEL_NAMES))
)

sample_output.to_csv(
    SHAP_RESULTS_DIR / "shap_test_sample.csv",
    index=False
)


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("\nSHAP analysis completed successfully.")
print(f"Outputs saved to:\n{SHAP_RESULTS_DIR}")