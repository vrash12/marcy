# build_training_dataset.py
"""
build_training_dataset.py
Exports questionnaire responses + final tech-field label to data/training_data.csv,
auto-detecting the correct response column and using a safe aggregation.
"""

import os
import pandas as pd
import mysql.connector

def build_dataset(csv_path="data/training_data.csv"):
    # 1) Connect to MySQL
    db = mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASS", ""),
        database=os.environ.get("DB_NAME", "ccsuggest"),
    )
    cursor = db.cursor()

    # 2) Introspect `responses` table for the answer column
    cursor.execute("SHOW COLUMNS FROM responses")
    cols = [row[0] for row in cursor.fetchall()]
    for candidate in ("answer", "response", "value", "response_value", "answer_text"):
        if candidate in cols:
            answer_col = candidate
            break
    else:
        raise RuntimeError(f"Could not find a response column in `responses`; found columns: {cols}")

    # 3) Query joined data
    query = f"""
        SELECT r.user_id,
               r.question_id,
               r.`{answer_col}` AS answer,
               rec.tech_field_id
        FROM responses r
        JOIN recommendations rec ON rec.user_id = r.user_id
    """
    df = pd.read_sql(query, db)

    # 4) Ensure answers are numeric if possible
    try:
        df["answer"] = pd.to_numeric(df["answer"])
    except ValueError:
        # leave as-is if non-numeric
        pass

    # 5) Pivot into feature matrix, using first() to avoid mean() on objects
    features = df.pivot_table(
        index="user_id",
        columns="question_id",
        values="answer",
        aggfunc="first"
    )
    features.columns = [f"Q{int(c)}" for c in features.columns]

    # 6) Extract the tech_field_id label per user
    labels = df.groupby("user_id")["tech_field_id"].first()

    # 7) Combine features + label, write CSV
    dataset = features.join(labels).reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    dataset.to_csv(csv_path, index=False)
    return dataset
