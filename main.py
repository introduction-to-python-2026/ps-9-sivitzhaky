
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1) Read config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

selected_features = cfg["selected_features"]
model_path = cfg["path"]

# 2) Load data
df = pd.read_csv("parkinsons.csv").dropna()

# 3) Build X,y using ONLY selected features
X = df[selected_features]
y = df["status"]

# 4) Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Pipeline + GridSearch (SVC)
pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svc", SVC(kernel="rbf"))
])

param_grid = {
    "svc__C": [0.1, 1, 10, 100, 1000],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
    "svc__class_weight": [None, "balanced"]
}

search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# 6) Validate
y_pred = best_model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print("Best params:", search.best_params_)
print("Validation accuracy:", val_acc)
