# dtree_service.py

from flask import Flask, request, jsonify
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib, os

app = Flask(__name__)
MODEL_PATH = "model.pkl"

def save_model(clf):
    joblib.dump(clf, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # if no model on disk yet, start fresh
    return DecisionTreeClassifier(criterion="entropy")

@app.post("/train")
def train():
    """
    Expects JSON:
    {
      "data": {
        "columns": ["Q1","Q2",…,"tech_field_id"],
        "data": [
          [3.0, 5.0, …, 2],
          [4.0, 1.0, …, 7],
          …
        ]
      }
    }
    """
    payload = request.json or {}
    matrix  = payload.get("data")
    if not matrix or "columns" not in matrix or "data" not in matrix:
        return jsonify({"error": "Invalid training payload"}), 400

    arr = np.array(matrix["data"])
    # last column is label
    X = arr[:, :-1].astype(float)
    y = arr[:,  -1].astype(int)

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)
    save_model(clf)
    return jsonify({"status": "trained"}), 200

@app.post("/retrain")
def retrain():
    #hello
    print("dsds")
    return train()

@app.post("/predict")
def predict():
    """
    Expects JSON: { "features": [3.0, 5.0, …] }
    Returns: { "1":0.72, "2":0.15, … }
    """
    clf = load_model()
    features = request.json.get("features", [])
    arr = np.array([features], dtype=float)
    proba = clf.predict_proba(arr)[0]
    classes = clf.classes_
    result = {int(c): float(p) for c, p in zip(classes, proba)}
    return jsonify(result)

if __name__ == "__main__":
    # listen on port 5000 by default
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
