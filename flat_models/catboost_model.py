import time
from catboost import CatBoostClassifier
from utils import load_data, evaluate_model, save_model_results

def main():
    X_train, X_test, y_train, y_test, labels, label_names = load_data()

    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        random_state=42,
        verbose=0
    )

    start_train = time.time()
    cat_model.fit(X_train, y_train)
    train_time = time.time() - start_train

    cat_results = evaluate_model(
        model_name="CatBoost",
        model=cat_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        label_names=label_names
    )

    save_model_results(
        model_name="CatBoost",
        results=cat_results,
        training_time=train_time
    )

if __name__ == "__main__":
    main()