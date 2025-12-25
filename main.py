import pandas as pd
df = pd.read_csv("parkinsons.csv")
df = df.dropna()
df.head()

import numpy as np
y = df["status"]
X = df.select_dtypes(include=[np.number]).drop(columns=["status"], errors="ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
pipe = Pipeline([("scaler", StandardScaler()),("svc", SVC(kernel="rbf"))])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
len(X_train), len(X_val), len(y_train), len(y_val)

from sklearn.model_selection import GridSearchCV

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

from sklearn.metrics import accuracy_score
best_model = search.best_estimator_
y_pred = best_model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)
print("Validation accuracy:", val_acc)
