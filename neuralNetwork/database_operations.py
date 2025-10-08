import sqlite3
import pandas as pd
import os

# 🌐 Ruta persistente para Hugging Face Spaces
DB_PATH = "/workspace/weather.db"
TABLE_NAME = "weather_data"

# ------------------ Conexión ------------------
def get_connection():
    """Crea y devuelve la conexión a SQLite, asegurando que la tabla exista."""
    # Crear carpeta /workspace si no existe
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Crear tabla si no existe
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
    conn.commit()
    return conn

# ------------------ Inserción ------------------
def insert_data(df: pd.DataFrame):
    """
    Inserta un DataFrame en la base de datos.
    El DataFrame debe tener las columnas:
    year, month, day, T2M_MAX, T2M_MIN, PRECTOTCORR, PRECSNO, WS2M_MAX
    """
    conn = get_connection()
    cursor = conn.cursor()

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
    """
    Actualiza filas según una condición.
    condition = {"year": 2025, "month": 1}
    updates = {"T2M_MAX": 25.0, "PRECTOTCORR": 0.5}
    """
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
def delete_data(condition: dict):
    """
    Elimina filas según una condición.
    condition = {"year": 2025, "month": 1}
    """
    conn = get_connection()
    cursor = conn.cursor()

    condition_str = " AND ".join([f"{k} = ?" for k in condition.keys()])
    sql = f"DELETE FROM {TABLE_NAME} WHERE {condition_str}"
    cursor.execute(sql, list(condition.values()))

    conn.commit()
    conn.close()
    print("✅ Datos eliminados correctamente")

# ------------------ Lectura ------------------
def read_data() -> pd.DataFrame:
    """Lee toda la tabla y devuelve un DataFrame de pandas."""
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df
