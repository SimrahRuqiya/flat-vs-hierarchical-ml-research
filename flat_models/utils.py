import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def load_data():
    """
    Load the preprocessed train/test datasets and apply SMOTE
    only to the training data.
    """
    X_train = pd.read_csv("../data/X_train.csv")
    X_test = pd.read_csv("../data/X_test.csv")

    y_train = pd.read_csv("../data/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/y_test.csv").squeeze()

    # Encoded labels
    labels = [0, 1, 2, 3, 4]

    # Human-readable labels
    label_names = [
        "Benign",
        "DDoS",
        "DoS",
        "Mirai",
        "Spoofing"
    ]

    print("\n--- Before SMOTE ---")
    print("Training Features :", X_train.shape)
    print("Testing Features  :", X_test.shape)
    print("Training Labels   :", y_train.shape)
    print("Testing Labels    :", y_test.shape)

    print("\nTraining Label Distribution:")
    print(y_train.value_counts().sort_index())

    # ---------------- Apply SMOTE ----------------
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    # Ensure compatibility across imbalanced-learn versions
    if not isinstance(X_train_resampled, pd.DataFrame):
        X_train_resampled = pd.DataFrame(
            X_train_resampled,
            columns=X_train.columns
        )

    if not isinstance(y_train_resampled, pd.Series):
        y_train_resampled = pd.Series(
            y_train_resampled,
            name=y_train.name
        )

    print("\n--- After SMOTE ---")
    print("Resampled Training Features :", X_train_resampled.shape)
    print("Resampled Training Labels   :", y_train_resampled.shape)

    print("\nResampled Training Label Distribution:")
    print(y_train_resampled.value_counts().sort_index())
    print("-----------------------------")

    # Keep the original test set untouched
    return (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
        labels,
        label_names
    )


def evaluate_model(model_name, model, X_test, y_test, labels, label_names):
    """
    Evaluate a trained model and save all outputs.
    """
    os.makedirs("../../results", exist_ok=True)

    # ---------------- Predictions ----------------
    start_pred = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred

    # Flatten predictions if necessary
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    # ---------------- Metrics ----------------
    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"\n{'=' * 60}")
    print(model_name)
    print("=" * 60)

    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")

    # ---------------- Classification Report ----------------
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nClassification Report")
    print(report_df)

    filename = model_name.lower().replace(" ", "_")

    report_df.to_csv(
        f"../results/{filename}_classification_report.csv"
    )

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names
    )

    cm_df.to_csv(
        f"../results/{filename}_confusion_matrix.csv"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        f"../results/{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ---------------- Return Summary ----------------
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Prediction Time": prediction_time
    }


def save_model_results(model_name, results, training_time):
    """
    Save the overall metrics for a model.
    """
    os.makedirs("../results", exist_ok=True)

    results["Training Time"] = training_time

    filename = model_name.lower().replace(" ", "_")

    pd.DataFrame([results]).to_csv(
        f"../results/{filename}_results.csv",
        index=False
    )

    print(f"\nSaved results to ../results/{filename}_results.csv")