import time
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, evaluate_model, save_model_results

def main():
    X_train, X_test, y_train, y_test, labels, label_names = load_data()

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    start_train = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_train

    rf_results = evaluate_model(
        model_name="Random Forest",
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        label_names=label_names
    )

    save_model_results(
        model_name="Random Forest",
        results=rf_results,
        training_time=train_time
    )

if __name__ == "__main__":
    main()