import time

from lightgbm import LGBMClassifier

from utils import load_data, evaluate_model, save_model_results


X_train, X_test, y_train, y_test, labels, label_names = load_data()

lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multiclass",
    num_class=len(labels),
    random_state=42,
    n_jobs=-1
)

start_train = time.time()
lgbm_model.fit(X_train, y_train)
train_time = time.time() - start_train

lgbm_results = evaluate_model(
    model_name="LightGBM",
    model=lgbm_model,
    X_test=X_test,
    y_test=y_test,
    labels=labels,
    label_names=label_names
)

save_model_results(
    model_name="LightGBM",
    results=lgbm_results,
    training_time=train_time
)