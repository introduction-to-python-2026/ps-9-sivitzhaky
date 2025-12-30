import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

FEATURES = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]

def main():
    # Load
    df = pd.read_csv("parkinsons.csv")

    # Keep only what the autograder will use
    df = df[FEATURES + ["status"]].dropna()
    X = df[FEATURES]
    y = df["status"]

    # Pipeline + GridSearch (refit=True by default -> fits best model on ALL X,y)
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

    search.fit(X, y)

    # Save EXACT filename used in config.yaml
    joblib.dump(search.best_estimator_, "my_model.joblib")


if __name__ == "__main__":
    main()
