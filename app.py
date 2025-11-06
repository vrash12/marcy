"""
Flask microservice exposing /predict and /retrain,
auto-building the CSV if missing.
"""

import os
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
from build_training_dataset import build_dataset
from dotenv import load_dotenv
from flask_cors import CORS
load_dotenv()

app        = Flask(__name__)
CORS(app, origins=["https://career-comm-main-laravel.onrender.com", "https://ccsuggest.netlify.app", "http://localhost:8000"])
MODEL_PATH = "model.pkl"
DATA_PATH  = "data/training_data.csv"
PDF_PATH   = "decision_tree.pdf"

REDIRECT_URL = os.environ.get("REDIRECT_URL", "http://127.0.0.1:8000")
@app.get("/")
def index():
    return render_template("index.html", redirect_url=REDIRECT_URL)

@app.get("/export_tree")
def export_tree():
    if os.path.exists(PDF_PATH):
        return send_file(PDF_PATH, as_attachment=True)
    return ("PDF not found", 404)

@app.get("/health")
def health():
    return {"status": "ok"}, 200

def ensure_csv():
    if not os.path.isfile(DATA_PATH):
        print("Training data CSV not found, building from database...")
        try:
            build_dataset(DATA_PATH)
            print(f"Training data built successfully: {DATA_PATH}")
        except Exception as e:
            print(f"Failed to build training dataset: {str(e)}")
            raise

def train_and_save():
    try:
        ensure_csv()
        df = pd.read_csv(DATA_PATH)
        
        # Check if we have sufficient data
        if df.empty:
            raise ValueError("Training dataset is empty. No responses found in database.")
        
        if len(df) < 10:
            raise ValueError(f"Insufficient training data: only {len(df)} records found. Need at least 10 records.")
        
        if "tech_field_id" not in df.columns:
            raise ValueError("Missing 'tech_field_id' column in training data")
        
        X = df.drop("tech_field_id", axis=1)
        y = df["tech_field_id"]
        
        # Check if we have feature columns
        if X.empty or len(X.columns) == 0:
            raise ValueError("No feature columns found in training data")
        
        print(f"Training with {len(df)} records and {len(X.columns)} features")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        dump(clf, MODEL_PATH)
        return clf
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

def get_model():
    return load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_and_save()

clf = get_model()

@app.post("/predict")
def predict():
    print("/predict payload:", request.json)
    feats = request.json.get("features", [])
    # Use DataFrame with feature names to avoid warning
    import pandas as pd
    df_feats = pd.DataFrame([feats], columns=clf.feature_names_in_)
    probs = clf.predict_proba(df_feats)[0]
    labels = clf.classes_.tolist()
    response = dict(zip(map(int, labels), probs.tolist()))
    print("\n\n/predict response:", response)
    return jsonify(response)

@app.post("/retrain")
def retrain():
    try:
        global clf
        clf = train_and_save()
        return jsonify({
            "status": "retrained", 
            "classes": clf.classes_.tolist(),
            "message": "Model retrained successfully"
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "suggestion": "Ensure you have completed questionnaires in the database before retraining"
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": f"Unexpected error during retraining: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=True
    )
