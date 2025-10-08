import sqlite3
import pandas as pd


DB_PATH = "/temp/weather.db"
TABLE_NAME = "weather_data"

# ------------------ Conexión ------------------
def get_connection():
    return sqlite3.connect(DB_PATH)

# ------------------ Inserción ------------------
def insert_data(df: pd.DataFrame):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            year INTEGER,
            month INTEGER,
            day INTEGER,
            T2M_MAX REAL,
            T2M_MIN REAL,
            PRECTOTCORR REAL,
            PRECSNO REAL,
            WS2M_MAX REAL
        )
    """)
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (year, month, day, T2M_MAX, T2M_MIN, PRECTOTCORR, PRECSNO, WS2M_MAX)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(row))
    conn.commit()
    conn.close()
    print("✅ Datos insertados correctamente")

# ------------------ Actualización ------------------
def update_data(condition: dict, updates: dict):
    conn = get_connection()
    cursor = conn.cursor()
    condition_str = " AND ".join([f"{k} = ?" for k in condition.keys()])
    updates_str = ", ".join([f"{k} = ?" for k in updates.keys()])
    sql = f"UPDATE {TABLE_NAME} SET {updates_str} WHERE {condition_str}"
    values = list(updates.values()) + list(condition.values())
    cursor.execute(sql, values)
    conn.commit()
    conn.close()
    print("✅ Datos actualizados correctamente")

# ------------------ Eliminación ------------------
def delete_data():
    conn = get_connection()
    cursor = conn.cursor()
    sql = f"DELETE FROM {TABLE_NAME}"
    cursor.execute(sql)
    conn.commit()
    conn.close()
    print("✅ Datos eliminados correctamente")

# ------------------ Lectura ------------------
def read_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df
