import time
from imblearn.ensemble import BalancedRandomForestClassifier
from utils import load_data, evaluate_model, save_model_results

def main():
    # This now loads the SMOTE-balanced training data automatically
    X_train, X_test, y_train, y_test, labels, label_names = load_data()

    # Use BalancedRandomForestClassifier for balanced dataset handling
    brf_model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    start_train = time.time()
    brf_model.fit(X_train, y_train)
    train_time = time.time() - start_train

    brf_results = evaluate_model(
        model_name="Balanced Random Forest", # Updated name for clarity in comparison
        model=brf_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        label_names=label_names
    )

    save_model_results(
        model_name="Balanced Random Forest", # Keeps file output naming aligned with compare_models.py
        results=brf_results,
        training_time=train_time
    )

if __name__ == "__main__":
    main()