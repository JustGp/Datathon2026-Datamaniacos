import json
import os
import re
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# IA
# =========================
Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=60.0,
    context_window=8000,
)

# =========================
# JSON DATABASE
# =========================
JSON_PATHS = [
    r"C:\Users\gaele\OneDrive\Desktop\Hackaton\clientes.json",
    r"C:\Users\gaele\OneDrive\Desktop\Hackaton\productos.json",
]

USER_ID_PATTERN = re.compile(r"^USR-\d{5}$", re.IGNORECASE)

# =========================
# HELPERS DE CARGA
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_df(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        if len(data) == 1:
            only_value = next(iter(data.values()))
            if isinstance(only_value, list):
                return pd.DataFrame(only_value)

        if all(not isinstance(v, (list, dict)) for v in data.values()):
            return pd.DataFrame([data])

        return pd.json_normalize(data)

    raise ValueError("Formato JSON no soportado.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ID" in df.columns and "user_id" not in df.columns:
        df = df.rename(columns={"ID": "user_id"})

    return df


def load_all() -> pd.DataFrame:
    dfs = []

    for path in JSON_PATHS:
        if not os.path.exists(path):
            print(f"Advertencia: no se encontro el archivo {path}")
            continue

        df = to_df(load_json(path))
        df = normalize_columns(df)
        df["__source_file__"] = path
        dfs.append(df)

    if not dfs:
        raise Exception("No se cargaron archivos JSON.")

    combined = pd.concat(dfs, ignore_index=True)

    if "user_id" not in combined.columns:
        raise Exception("La columna 'user_id' no existe en los JSON cargados.")

    return combined


CLIENTS_DF = load_all()

# =========================
# MEMORIA DE SESIONES
# =========================
sessions: dict[str, str] = {}


def get_user(session_id: str) -> str | None:
    return sessions.get(session_id)


def save_user(session_id: str, user_id: str) -> None:
    sessions[session_id] = user_id


def is_valid_id(user_input: str) -> bool:
    return bool(USER_ID_PATTERN.fullmatch(str(user_input).strip().upper()))

# =========================
# HELPERS DE DATOS
# =========================
def get_user_rows(user_id: str) -> pd.DataFrame:
    return CLIENTS_DF[CLIENTS_DF["user_id"].astype(str).str.upper() == str(user_id).upper()]


def get_first_non_null_value(rows: pd.DataFrame, column_names: list[str]):
    for column_name in column_names:
        if column_name in rows.columns:
            values = rows[column_name].dropna()
            if not values.empty:
                return values.iloc[0]
    return None


def get_user_balance(user_id: str):
    rows = get_user_rows(user_id)
    if rows.empty:
        return None

    return get_first_non_null_value(
        rows,
        ["saldo", "balance", "saldo_actual", "available_balance", "monto_saldo"],
    )


def get_user_products(user_id: str) -> list[str]:
    rows = get_user_rows(user_id)
    if rows.empty:
        return []

    possible_columns = [
        "producto",
        "productos",
        "product_name",
        "nombre_producto",
        "tipo_producto",
        "producto_bancario",
    ]

    products = []
    for col in possible_columns:
        if col in rows.columns:
            values = rows[col].dropna().astype(str).tolist()
            products.extend(values)

    # Si no hay columna explícita de producto, intenta derivarlo
    if not products:
        derived_products = []

        if "num_productos_activos" in rows.columns:
            count_value = get_first_non_null_value(rows, ["num_productos_activos"])
            if count_value is not None:
                derived_products.append(f"Productos activos registrados: {count_value}")

        if "es_hey_pro" in rows.columns:
            hey_pro = get_first_non_null_value(rows, ["es_hey_pro"])
            if hey_pro is True:
                derived_products.append("Hey Pro")

        if "tiene_seguro" in rows.columns:
            seguro = get_first_non_null_value(rows, ["tiene_seguro"])
            if seguro is True:
                derived_products.append("Seguro")

        products.extend(derived_products)

    unique_products = []
    seen = set()
    for product in products:
        clean_product = str(product).strip()
        if clean_product and clean_product not in seen:
            seen.add(clean_product)
            unique_products.append(clean_product)

    return unique_products


def get_account_summary(user_id: str) -> dict[str, Any] | None:
    rows = get_user_rows(user_id)
    if rows.empty:
        return None

    record = rows.iloc[0].to_dict()

    summary = {
        "user_id": record.get("user_id"),
        "saldo": record.get("saldo"),
        "ingreso_mensual_mxn": record.get("ingreso_mensual_mxn"),
        "num_productos_activos": record.get("num_productos_activos"),
        "estado": record.get("estado"),
        "ciudad": record.get("ciudad"),
        "score_buro": record.get("score_buro"),
        "ocupacion": record.get("ocupacion"),
    }

    return summary


def get_available_fields() -> list[str]:
    return [col for col in CLIENTS_DF.columns if col != "__source_file__"]


def ask_llm_general(user_id: str, mensaje: str, user_rows: pd.DataFrame) -> str:
    user_data = user_rows.to_dict(orient="records")

    prompt = f"""
Eres un asistente bancario.

USUARIO AUTENTICADO:
ID: {user_id}

DATOS REALES DEL USUARIO:
{user_data}

PREGUNTA:
{mensaje}

Reglas:
- Responde solo con base en los datos reales del usuario cuando la pregunta trate sobre sus datos.
- Si el dato no existe en los datos reales, dilo claramente.
- No inventes informacion.
- Si la pregunta es general, responde de forma breve y clara.
- Responde en espanol.
"""

    response = Settings.llm.complete(prompt)
    return str(response)

# =========================
# ENDPOINTS
# =========================
@app.post("/chat")
async def chat(data: dict):
    mensaje = (data.get("mensaje") or "").strip()
    session_id = data.get("session_id", "default")

    if not mensaje:
        return {"respuesta": "Envia un mensaje valido."}

    user_id = get_user(session_id)

    if not user_id:
        if not is_valid_id(mensaje):
            return {"respuesta": "ID invalida. Debe tener formato USR-00001."}

        clean_user_id = mensaje.upper()
        save_user(session_id, clean_user_id)
        return {"respuesta": f"ID registrada ({clean_user_id}). Ahora dime en que puedo ayudarte."}

    user_rows = get_user_rows(user_id)
    if user_rows.empty:
        return {"respuesta": f"No se encontraron datos para el ID {user_id}."}

    mensaje_lower = mensaje.lower()

    try:
        if "saldo" in mensaje_lower:
            saldo = get_user_balance(user_id)
            if saldo is None:
                return {"respuesta": "No encontre informacion de saldo para este usuario."}
            return {"respuesta": f"Tu saldo actual es {saldo} MXN."}

        if "estado de cuenta" in mensaje_lower or "estado cuenta" in mensaje_lower:
            summary = get_account_summary(user_id)
            if summary is None:
                return {"respuesta": "No encontre informacion de estado de cuenta para este usuario."}

            respuesta = (
                f"Este es tu estado de cuenta resumido:\n"
                f"- ID: {summary.get('user_id')}\n"
                f"- Saldo: {summary.get('saldo')} MXN\n"
                f"- Ingreso mensual: {summary.get('ingreso_mensual_mxn')} MXN\n"
                f"- Productos activos: {summary.get('num_productos_activos')}\n"
                f"- Estado: {summary.get('estado')}\n"
                f"- Ciudad: {summary.get('ciudad')}\n"
                f"- Score buro: {summary.get('score_buro')}\n"
                f"- Ocupacion: {summary.get('ocupacion')}"
            )
            return {"respuesta": respuesta}

        if (
            "productos" in mensaje_lower
            or "producto bancario" in mensaje_lower
            or "mis productos" in mensaje_lower
        ):
            products = get_user_products(user_id)
            if not products:
                return {"respuesta": "No encontre productos bancarios registrados para este usuario."}
            return {"respuesta": "Tus productos bancarios son: " + ", ".join(products)}

        if "que informacion" in mensaje_lower or "campos disponibles" in mensaje_lower:
            fields = get_available_fields()
            return {"respuesta": "Puedo consultar estos campos: " + ", ".join(fields)}

        respuesta = ask_llm_general(user_id, mensaje, user_rows)
        return {"respuesta": respuesta}

    except Exception as e:
        return {"respuesta": f"Error interno del servidor: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "ok"}
