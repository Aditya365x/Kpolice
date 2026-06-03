"""
STEP 1: Drops old tables
STEP 2: Creates new tables
STEP 3: Loads all FIR data from JSON

Usage:
  pip install psycopg2-binary
  python setup_and_load.py
"""

import json
import os
import psycopg2
from psycopg2.extras import execute_values

# ─── CHANGE THESE ─────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "kpolice",
    "user":     "postgres",
    
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "fir_20_instances.json")
# ──────────────────────────────────────────────────────────


DROP_SQL = """
DROP TABLE IF EXISTS evidence      CASCADE;
DROP TABLE IF EXISTS fir_accused   CASCADE;
DROP TABLE IF EXISTS fir_victim    CASCADE;
DROP TABLE IF EXISTS fir           CASCADE;
DROP TABLE IF EXISTS person        CASCADE;
DROP TABLE IF EXISTS police_station CASCADE;
"""


CREATE_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE police_station (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    district    VARCHAR(100) NOT NULL,
    state       VARCHAR(50)  DEFAULT 'Karnataka',
    lat         DECIMAL(9,6),
    lng         DECIMAL(9,6)
);

CREATE TABLE person (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150),
    age             INT,
    gender          VARCHAR(10),
    address         TEXT,
    phone           VARCHAR(20),
    is_withheld     BOOLEAN DEFAULT FALSE
);

CREATE TABLE fir (
    id              SERIAL PRIMARY KEY,
    fir_number      VARCHAR(20) UNIQUE NOT NULL,
    station_id      INT REFERENCES police_station(id),
    filed_at        TIMESTAMP,
    occurred_at     TIMESTAMP,
    crime_type      VARCHAR(100),
    ipc_sections    TEXT[],
    description     TEXT,
    location_text   VARCHAR(255),
    lat             DECIMAL(9,6),
    lng             DECIMAL(9,6),
    status          VARCHAR(30),
    io_name         VARCHAR(100),
    court_case_no   VARCHAR(50),
    complainant_id  INT REFERENCES person(id)
);

CREATE TABLE fir_victim (
    id          SERIAL PRIMARY KEY,
    fir_id      INT REFERENCES fir(id) ON DELETE CASCADE,
    person_id   INT REFERENCES person(id),
    injury_desc TEXT
);

CREATE TABLE fir_accused (
    id              SERIAL PRIMARY KEY,
    fir_id          INT REFERENCES fir(id) ON DELETE CASCADE,
    person_id       INT REFERENCES person(id),
    is_known        BOOLEAN DEFAULT FALSE,
    physical_desc   TEXT,
    is_arrested     BOOLEAN DEFAULT FALSE
);

CREATE TABLE evidence (
    id          SERIAL PRIMARY KEY,
    fir_id      INT REFERENCES fir(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    type        VARCHAR(20)
);
"""


# ─── HELPERS ──────────────────────────────────────────────

def get_or_create_station(cur, s):
    cur.execute(
        "SELECT id FROM police_station WHERE name=%s AND district=%s",
        (s["name"], s["district"])
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO police_station (name, district, lat, lng) VALUES (%s,%s,%s,%s) RETURNING id",
        (s["name"], s["district"], s["lat"], s["lng"])
    )
    return cur.fetchone()[0]


def insert_person(cur, name, age, gender, address, phone):
    withheld = any(w in str(name or "") for w in ["[withheld]", "[Identity withheld", "identity withheld"])
    cur.execute(
        "INSERT INTO person (name, age, gender, address, phone, is_withheld) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
        (name, age, gender, address, phone, withheld)
    )
    return cur.fetchone()[0]


def classify_evidence(desc):
    d = desc.lower()
    if any(w in d for w in ["cctv", "footage", "recording", "call record", "gps", "cell tower", "breathalyzer"]):
        return "digital"
    if any(w in d for w in ["report", "document", "agreement", "extract", "note", "receipt", "papers", "form"]):
        return "document"
    if any(w in d for w in ["witness", "statement", "eyewitness"]):
        return "witness"
    return "physical"


# ─── MAIN ─────────────────────────────────────────────────

def run():
    with open(JSON_FILE, encoding="utf-8") as f:
        firs = json.load(f)["firs"]

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Step 1: Drop
    print("Dropping old tables...")
    cur.execute(DROP_SQL)
    conn.commit()
    print("Done.")

    # Step 2: Create
    print("Creating new tables...")
    cur.execute(CREATE_SQL)
    conn.commit()
    print("Done.")

    # Step 3: Load
    print(f"\nLoading {len(firs)} FIR records...\n")
    ok = 0
    fail = 0

    for fir in firs:
        try:
            # Station
            station_id = get_or_create_station(cur, fir["station"])

            # Complainant
            c = fir["complainant"]
            complainant_id = insert_person(cur,
                c.get("name"), c.get("age"), c.get("gender"), None, c.get("phone")
            )

            # FIR
            cur.execute("""
                INSERT INTO fir (
                    fir_number, station_id, filed_at, occurred_at,
                    crime_type, ipc_sections, description,
                    location_text, lat, lng,
                    status, io_name, court_case_no, complainant_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                fir["fir_number"], station_id,
                fir["filed_at"], fir.get("occurred_at"),
                fir["crime_type"], fir["ipc_sections"],
                fir["description"], fir.get("location_text"),
                fir["lat"], fir["lng"],
                fir["status"], fir.get("io"),
                fir.get("court_case_no"), complainant_id
            ))
            fir_db_id = cur.fetchone()[0]

            # Victim
            v = fir.get("victim", {})
            if v and v.get("name"):
                victim_person_id = insert_person(cur,
                    v.get("name"), v.get("age"), v.get("gender"), v.get("address"), None
                )
                cur.execute(
                    "INSERT INTO fir_victim (fir_id, person_id, injury_desc) VALUES (%s,%s,%s)",
                    (fir_db_id, victim_person_id, v.get("injury"))
                )

            # Accused
            for acc in fir.get("accused", []):
                is_arrested = "arrested" in str(acc.get("physical_desc", "")).lower()
                acc_person_id = insert_person(cur,
                    acc.get("name"), acc.get("age"), None, acc.get("address"), None
                )
                cur.execute("""
                    INSERT INTO fir_accused (fir_id, person_id, is_known, physical_desc, is_arrested)
                    VALUES (%s,%s,%s,%s,%s)
                """, (fir_db_id, acc_person_id, acc.get("is_known", False),
                      acc.get("physical_desc"), is_arrested))

            # Evidence
            ev_rows = [
                (fir_db_id, desc, classify_evidence(desc))
                for desc in fir.get("evidence", [])
            ]
            if ev_rows:
                execute_values(cur,
                    "INSERT INTO evidence (fir_id, description, type) VALUES %s",
                    ev_rows
                )

            conn.commit()
            ok += 1
            print(f"  ✓  {fir['fir_number']}  —  {fir['crime_type']}")

        except Exception as e:
            conn.rollback()
            fail += 1
            print(f"  ✗  {fir['fir_number']}  FAILED: {e}")

    cur.close()
    conn.close()
    print(f"\n{'='*40}")
    print(f"  Loaded : {ok}")
    print(f"  Failed : {fail}")
    print(f"{'='*40}")


if __name__ == "__main__":
    run()