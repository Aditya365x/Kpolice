import os
import json
import re
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

SUPABASE_HOST = "db.qvrqdlqzwtiormcaimvw.supabase.co"
SUPABASE_PORT = 5432
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres"
SUPABASE_PASSWORD = "tsj%%638ttTTY**!@"
SUPABASE_SSLMODE = "require"

DEEPSEEK_API_KEY = "sk-e3490f67707e492497b5ba033f8d35f1"

SCHEMA = """CREATE TABLE fir_records (
  id INTEGER, fir_id INTEGER, fir_number TEXT, gd_reference TEXT,
  station_name TEXT, station_district TEXT, station_lat FLOAT, station_lng FLOAT,
  filed_at TIMESTAMP, from_time TIMESTAMP, to_time TIMESTAMP,
  delay_in_reporting_hours INTEGER, reason_for_delay TEXT,
  information_type TEXT, crime_type TEXT, description TEXT,
  charges_data JSONB, location_text TEXT, location_distance_km FLOAT,
  location_direction TEXT, location_lat FLOAT, location_lng FLOAT,
  status TEXT, people_data JSONB, investigation_data JSONB, court_case_no TEXT
);"""


def query_deepseek(prompt: str) -> str:
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a PostgreSQL expert. Convert questions to SQL queries."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def get_db_connection():
    return psycopg2.connect(
        host=SUPABASE_HOST,
        port=SUPABASE_PORT,
        dbname=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        sslmode="require",
    )

def nl_to_sql(question: str) -> str:
    prompt = f"""You are a PostgreSQL expert. Given this table schema:

{SCHEMA}

Convert this question into a PostgreSQL SQL query. Return ONLY the SQL query, nothing else.

Question: {question}
SQL:"""
    response = query_deepseek(prompt)
    if not response.strip():
        raise ValueError("Empty response from DeepSeek API")
    sql = response.strip()
    sql = re.sub(r"^```sql\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    return sql

def execute_sql(sql: str) -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()

def format_results(rows: list[dict]) -> str:
    if not rows:
        return "No results found."
    result = f"Returned {len(rows)} row(s):\n"
    for i, row in enumerate(rows[:20]):
        result += f"{i + 1}. {json.dumps(row, indent=2, default=str)}\n"
    if len(rows) > 20:
        result += f"... and {len(rows) - 20} more rows."
    return result

def chat():
    print("=" * 60)
    print("  FIR Records Text-to-SQL Chatbot (DeepSeek)")
    print("  Type 'exit' to quit")
    print("=" * 60)
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if not question:
            continue
        try:
            print("\n[Calling DeepSeek...]", flush=True)
            sql = nl_to_sql(question)
            print(f"SQL: {sql}")
            print("[Executing on Supabase...]", flush=True)
            rows = execute_sql(sql)
            print(format_results(rows))
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    chat()
