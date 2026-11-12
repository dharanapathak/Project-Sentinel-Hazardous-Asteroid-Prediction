from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sample_n = min(25000, len(X_train))
    if sample_n < len(X_train):
        X_cv, _, y_cv, _ = train_test_split(
            X_train, y_train, train_size=sample_n, stratify=y_train,
            random_state=RANDOM_STATE
        )
    else:
        X_cv, y_cv = X_train, y_train

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                     solver="liblinear", random_state=RANDOM_STATE))
    ])
    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=1))
    ])
    scoring = {"f1":"f1", "recall":"recall", "precision":"precision", "roc_auc":"roc_auc"}
    comparison = {}
    for name, pipe in [("Logistic Regression (balanced)", logistic),
                       ("Random Forest (balanced)", rf)]:
        scores = cross_validate(pipe, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=1)
        comparison[name] = {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}

    search = RandomizedSearchCV(
        Pipeline([("preprocessor", preprocessor),
                  ("model", RandomForestClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=1))]),
        param_distributions={
            "model__n_estimators":[50, 80],
            "model__max_depth":[None, 18],
            "model__min_samples_leaf":[1, 3],
            "model__max_features":["sqrt", 0.8]
        },
        n_iter=2, scoring="f1", cv=2, random_state=RANDOM_STATE,
        n_jobs=1, refit=True
    )
    search.fit(X_cv, y_cv)
    logistic_f1 = comparison["Logistic Regression (balanced)"]["test_f1"]
    if search.best_score_ >= logistic_f1:
        final_model = search.best_estimator_
        selected = "Tuned Random Forest (balanced)"
    else:
        final_model = logistic
        selected = "Logistic Regression (balanced)"
    final_model.fit(X_train, y_train)

    pred = final_model.predict(X_test)
    proba = final_model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_hazardous": float(precision_score(y_test, pred, zero_division=0)),
        "recall_hazardous": float(recall_score(y_test, pred, zero_division=0)),
        "f1_hazardous": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba))
    }
    details = {
        "selected_model": selected,
        "comparison": comparison,
        "best_random_forest_cv_f1": float(search.best_score_),
        "best_random_forest_params": search.best_params_,
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "test_rows": int(len(X_test))
    }
    return final_model, details

def predict_one(model, feature_values: dict, feature_columns: list[str]):
    missing = [c for c in feature_columns if c not in feature_values]
    if missing:
        raise ValueError("Missing input feature(s): " + ", ".join(missing))
    frame = pd.DataFrame([{c: feature_values[c] for c in feature_columns}])
    pred = int(model.predict(frame)[0])
    score = float(np.asarray(model.predict_proba(frame))[0, 1]) if hasattr(model, "predict_proba") else None
    return {"prediction": "Hazardous" if pred == 1 else "Non-Hazardous",
            "hazardous_class_probability_score": score}

def save_pipeline(model, details, path):
    joblib.dump({"pipeline": model, "metadata": details}, path)

def load_pipeline(path):
    return joblib.load(path)
