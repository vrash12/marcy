"""
Flask microservice exposing /predict and /retrain,
auto-building the CSV if missing. Cloud Run friendly.
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

app = Flask(__name__)

# --- Cloud Run: read-only FS. Use /tmp for anything written at runtime. ---
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/tmp")
os.makedirs(STORAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(STORAGE_DIR, "model.pkl")
DATA_PATH  = os.path.join(STORAGE_DIR, "training_data.csv")
PDF_PATH   = os.path.join(STORAGE_DIR, "decision_tree.pdf")  # optional

# CORS domains, comma-separated
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS",
        "https://career-comm-main-laravel.onrender.com,https://ccsuggest.netlify.app,http://localhost:8000"
    ).split(",")
    if o.strip()
]
CORS(app, origins=CORS_ORIGINS)

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

@app.get("/features")
def features():
    """Expose feature column names expected by the model (helps your frontend)."""
    try:
        model = get_model()
        return jsonify({"features": model.feature_names_in_.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def ensure_csv():
    if not os.path.isfile(DATA_PATH):
        app.logger.info("Training data CSV not found, building from database...")
        build_dataset(DATA_PATH)
        app.logger.info(f"Training data built successfully: {DATA_PATH}")

def train_and_save():
    try:
        ensure_csv()
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Training dataset is empty. No responses found in database.")
        if len(df) < 10:
            raise ValueError(f"Insufficient training data: only {len(df)} records found. Need at least 10 records.")
        if "tech_field_id" not in df.columns:
            raise ValueError("Missing 'tech_field_id' column in training data")

        X = df.drop("tech_field_id", axis=1)
        y = df["tech_field_id"]

        if X.empty or len(X.columns) == 0:
            raise ValueError("No feature columns found in training data")

        app.logger.info(f"Training with {len(df)} records and {len(X.columns)} features")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        dump(clf, MODEL_PATH)
        return clf
    except Exception as e:
        app.logger.exception(f"Training failed: {str(e)}")
        raise

def get_model():
    return load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_and_save()

# Load model at startup (will also build the CSV if missing)
clf = get_model()

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    feats = payload.get("features", [])

    # Validate length early
    expected = len(clf.feature_names_in_)
    if len(feats) != expected:
        return jsonify({
            "status": "error",
            "error": f"Expected {expected} features in this order: {clf.feature_names_in_.tolist()}",
        }), 400

    df_feats = pd.DataFrame([feats], columns=clf.feature_names_in_)
    probs = clf.predict_proba(df_feats)[0]
    labels = clf.classes_.tolist()
    response = dict(zip(list(map(int, labels)), probs.tolist()))
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
