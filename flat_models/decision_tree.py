import time
from sklearn.tree import DecisionTreeClassifier
from utils import load_data, evaluate_model, save_model_results

def main():
    X_train, X_test, y_train, y_test, labels, label_names = load_data()

    dt_model = DecisionTreeClassifier(random_state=42)

    start_train = time.time()
    dt_model.fit(X_train, y_train)
    train_time = time.time() - start_train

    dt_results = evaluate_model(
        model_name="Decision Tree",
        model=dt_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        label_names=label_names
    )

    save_model_results(
        model_name="Decision Tree",
        results=dt_results,
        training_time=train_time
    )

if __name__ == "__main__":
    main()