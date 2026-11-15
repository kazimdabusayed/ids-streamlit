"""Core logic for the Intrusion Detection (NSL-KDD) Streamlit app.

Loads the ARFF files, preprocesses them, trains three classifiers
(Decision Tree, Random Forest, Naive Bayes) and provides prediction helpers.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
CATS = ["protocol_type", "service", "flag"]
CACHE = BASE / "ids_model_cache.joblib"
CACHE_VERSION = 1


def load_arff(path):
    """Read an ARFF file into a DataFrame (regex-based header parsing)."""
    cols, data_start = [], 0
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+'([^']+)'", line)
            if m:
                cols.append(m.group(1))
        if line.lower().startswith("@data"):
            data_start = i + 1
            break
    return pd.read_csv(path, skiprows=data_start, header=None, names=cols)


def find_data_files(folder=BASE):
    train = sorted(Path(folder).glob("KDDTrain*.arff"))
    test = sorted(Path(folder).glob("KDDTest*.arff"))
    if not train or not test:
        raise FileNotFoundError(
            "Could not find KDDTrain_.arff and KDDTest_.arff next to app.py. "
            "Copy both ARFF files into the same folder as app.py and restart."
        )
    return train[0], test[0]


def build_bundle(train_path, test_path):
    tr = load_arff(train_path)
    te = load_arff(test_path)
    stats = {
        "Train records (raw)": len(tr),
        "Test records (raw)": len(te),
        "Duplicates removed (train)": int(tr.duplicated().sum()),
        "Duplicates removed (test)": int(te.duplicated().sum()),
        "Missing values": int(tr.isnull().sum().sum() + te.isnull().sum().sum()),
    }
    tr = tr.drop_duplicates().reset_index(drop=True)
    te = te.drop_duplicates().reset_index(drop=True)
    test_raw = te.copy()  # keeps readable text values for the UI

    encoders = {}
    for c in CATS:
        le = LabelEncoder().fit(pd.concat([tr[c], te[c]]))
        encoders[c] = le
        tr[c] = le.transform(tr[c])
        te[c] = le.transform(te[c])
    label_encoder = LabelEncoder().fit(pd.concat([tr["class"], te["class"]]))
    tr["class"] = label_encoder.transform(tr["class"])
    te["class"] = label_encoder.transform(te["class"])

    feats = [c for c in tr.columns if c != "class"]
    X_train, y_train = tr[feats], tr["class"]
    X_test, y_test = te[feats], te["class"]

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }
    results, cms, reports = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        cms[name] = confusion_matrix(y_test, pred)
        reports[name] = classification_report(
            y_test, pred, target_names=label_encoder.classes_, zero_division=0)
    results_df = pd.DataFrame(results).T

    class_dist = pd.DataFrame({
        "Train": pd.Series(label_encoder.inverse_transform(y_train)).value_counts(),
        "Test": pd.Series(label_encoder.inverse_transform(y_test)).value_counts(),
    })
    return {
        "version": CACHE_VERSION, "feats": feats, "encoders": encoders,
        "label_encoder": label_encoder, "models": models, "results_df": results_df,
        "best": results_df["Accuracy"].idxmax(), "cms": cms, "reports": reports,
        "X_test": X_test, "y_test": y_test, "test_raw": test_raw,
        "stats": stats, "class_dist": class_dist,
    }


def get_bundle():
    """Load the trained bundle from disk, or train it (first run, ~1 minute)."""
    if CACHE.exists():
        try:
            b = joblib.load(CACHE)
            if b.get("version") == CACHE_VERSION:
                return b
        except Exception:
            pass  # corrupted / incompatible cache -> retrain
    train_path, test_path = find_data_files()
    b = build_bundle(train_path, test_path)
    joblib.dump(b, CACHE, compress=3)
    return b


def encode_row(values, bundle):
    """values: {feature: raw value (text for categorical columns)} -> 1-row DataFrame."""
    row = {}
    for f in bundle["feats"]:
        if f in bundle["encoders"]:
            row[f] = int(bundle["encoders"][f].transform([str(values[f])])[0])
        else:
            row[f] = float(values[f])
    return pd.DataFrame([row])[bundle["feats"]]


def predict_all(bundle, X):
    """Run every model on a 1-row DataFrame."""
    classes = list(bundle["label_encoder"].classes_)
    out = {}
    for name, model in bundle["models"].items():
        proba = model.predict_proba(X)[0]
        label = classes[int(model.predict(X)[0])]
        out[name] = {"Prediction": label,
                     **{f"P({c})": round(float(p), 4) for c, p in zip(classes, proba)}}
    return out


def validate_rows(bundle, model_name, indices):
    """Actual vs predicted for chosen test-set rows."""
    classes = bundle["label_encoder"].classes_
    X = bundle["X_test"].iloc[indices]
    pred = bundle["models"][model_name].predict(X)
    actual = bundle["y_test"].iloc[indices].to_numpy()
    df = pd.DataFrame({
        "Row": indices,
        "Actual": [classes[int(a)] for a in actual],
        "Predicted": [classes[int(p)] for p in pred],
    })
    df["Match"] = np.where(df["Actual"] == df["Predicted"], "✅", "❌")
    return df
