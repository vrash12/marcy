"""
seed_training_data.py
--------------------------------------------
1. Fetch *real* question metadata (id, type, tech_field_id) from MySQL.
2. Create N synthetic students whose answers respect:
   • question.type  – 'scale', 'single', or 'multiple'
   • embedded bias  – answers tied to their eventual tech_field_id
3. Write the result to data/training_data.csv for model training.
"""

import os, random, csv, itertools, pathlib, mysql.connector

# ------------------------------------------------------------------
N_ROWS       = 800                        # how many synthetic students
NUM_OPTIONS  = 4                          # assume 4 options for single / multiple
SCALE_RANGE  = range(1, 6)                # 1–5 Likert scale
CSV_PATH     = "data/training_data.csv"
# ------------------------------------------------------------------

def db_conn():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 3307)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "ccsuggest"),
    )

def fetch_questions():
    """Return list of (id, type, tech_field_id_or_None) ordered by id."""
    with db_conn() as db:
        cur = db.cursor()
        cur.execute(
            "SELECT id, type, tech_field_id FROM questions ORDER BY id"
        )
        rows = cur.fetchall()
    return rows

# -----------------------  answer generators ------------------------

def gen_scale(correlated: bool) -> int:
    """Higher values if correlated (4–5), else uniform 1–5."""
    return random.randint(4, 5) if correlated else random.choice(SCALE_RANGE)

def gen_single(correlated: bool) -> int:
    """
    Single-choice encoded as 0–(NUM_OPTIONS-1).
    We bias to option 0 when correlated, else random.
    """
    return 0 if correlated else random.randint(0, NUM_OPTIONS - 1)

def gen_multiple(correlated: bool) -> int:
    """
    Multiple-choice -> multi-hot bit-mask in an int.
    bit i = 1 means option i selected.
    If correlated, always include option 0; otherwise random subset.
    """
    # always include option 0 for correlation
    base = 1 if correlated else 0
    # randomly select remaining bits
    for i in range(1, NUM_OPTIONS):
        if random.random() < 0.25:          # 25 % chance per option
            base |= (1 << i)
    return base

# ------------------------  main synthesiser ------------------------

def build_dataset():
    q_meta = fetch_questions()  # [(id, type, tech_field_id), …]
    columns = [f"Q{qid}" for qid, _, _ in q_meta] + ["tech_field_id"]

    pathlib.Path("data").mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)

        for _ in range(N_ROWS):
            # assign a target tech_field for the synthetic student
            tech_field = random.randint(1, 11)
            row = []

            for qid, qtype, q_field in q_meta:
                correlated = (q_field == tech_field)  # bias if same field
                if qtype == "scale":
                    row.append(gen_scale(correlated))
                elif qtype == "single":
                    row.append(gen_single(correlated))
                else:  # 'multiple'
                    row.append(gen_multiple(correlated))
            row.append(tech_field)
            writer.writerow(row)

    print(f"✅ Generated {N_ROWS} rows ➜ {CSV_PATH}")

if __name__ == "__main__":
    build_dataset()
