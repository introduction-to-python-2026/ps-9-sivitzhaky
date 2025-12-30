import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# The autograder expects the model to use exactly the 2 features listed in config.yaml:
FEATURES = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]


def main():
    # 1) Load dataset
    df = pd.read_csv("parkinsons.csv")

    # 2) Keep only required columns + label, and drop missing rows
    required_cols = FEATURES + ["status"]
    df = df[required_cols].dropna()

    X = df[FEATURES]
    y = df["status"]

    # 3) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Model pipeline (scaling + SVC), then hyperparameter search
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf")),
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        "svc__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    # 5) Save the best model exactly where config.yaml points
    best_model = search.best_estimator_
    joblib.dump(best_model, "my_model.joblib")


if __name__ == "__main__":
    main()
