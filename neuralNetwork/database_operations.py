import os
import sqlite3
from pathlib import Path
import pandas as pd

# 🌐 Ruta persistente para base de datos SQLite (compatible con local, Windows, Linux y HF Spaces)
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Usar variable de entorno si existe, o por defecto workspace/weather.db
DEFAULT_DB = str(WORKSPACE_DIR / "weather.db")
DB_PATH = os.environ.get("DB_PATH", DEFAULT_DB)
TABLE_NAME = "weather_data"


# ------------------ Conexión ------------------
def get_connection():
    """Crea y devuelve la conexión a SQLite, asegurando que el directorio y la tabla existan."""
    db_file = Path(DB_PATH)
    try:
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_file))
    except Exception:
        # Fallback si no se puede escribir en DB_PATH (ej. /tmp en Windows)
        fallback_path = WORKSPACE_DIR / "weather.db"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(fallback_path))

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
    conn.commit()
    return conn


# ------------------ Inserción ------------------
def insert_data(df: pd.DataFrame):
    """
    Inserta un DataFrame en la base de datos.
    El DataFrame debe tener las columnas:
    year, month, day, T2M_MAX, T2M_MIN, PRECTOTCORR, PRECSNO, WS2M_MAX
    """
    if df.empty:
        return

    conn = get_connection()
    cursor = conn.cursor()

    columns = ["year", "month", "day", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "PRECSNO", "WS2M_MAX"]
    df_subset = df[[col for col in columns if col in df.columns]]

    for _, row in df_subset.iterrows():
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (year, month, day, T2M_MAX, T2M_MIN, PRECTOTCORR, PRECSNO, WS2M_MAX)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(row.get(col, 0.0) for col in columns))

    conn.commit()
    conn.close()


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


# ------------------ Eliminación ------------------
def delete_data():
    conn = get_connection()
    cursor = conn.cursor()
    sql = f"DELETE FROM {TABLE_NAME}"
    cursor.execute(sql)
    conn.commit()
    conn.close()


# ------------------ Lectura ------------------
def read_data() -> pd.DataFrame:
    """Lee toda la tabla y devuelve un DataFrame de pandas."""
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df
