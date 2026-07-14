import time
from xgboost import XGBClassifier
from utils import load_data, evaluate_model, save_model_results

def main():
    X_train, X_test, y_train, y_test, labels, label_names = load_data()

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        num_class=len(labels),
        random_state=42,
        n_jobs=-1
    )

    start_train = time.time()
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_train

    xgb_results = evaluate_model(
        model_name="XGBoost",
        model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        label_names=label_names
    )

    save_model_results(
        model_name="XGBoost",
        results=xgb_results,
        training_time=train_time
    )

if __name__ == "__main__":
    main()