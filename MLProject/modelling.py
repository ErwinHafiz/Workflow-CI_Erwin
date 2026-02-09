import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub





# LOAD DATA
df = pd.read_csv("dataset_preprocessing/titanic_clean.csv")

X = df.drop("2urvived", axis=1)
y = df["2urvived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()
# MLFLOW EXPERIMENT
# mlflow.set_experiment("Titanic-Baseline")

# with mlflow.start_run():
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# preds = model.predict(X_test)
# acc = accuracy_score(y_test, preds)
# LOGGING
# mlflow.log_metric("accuracy", acc)
# mlflow.sklearn.log_model(model, "model")






import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import sys

# import dagshub
# INIT DAGSHUB + MLFLOW
# if os.getenv("CI") != "true":
#     dagshub.init(
#         repo_owner="erwinhafizzxr",
#         repo_name="titanic-mlflow-erwin-hafiz-triadi",
#         mlflow=True
#     )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # use param MLflow 
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    # read data
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "dataset_preprocessing/titanic_clean.csv"
    df = pd.read_csv(dataset_path)

    X = df.drop("2urvived", axis=1)
    y = df["2urvived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )
