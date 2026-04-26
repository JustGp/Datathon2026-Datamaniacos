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
    r"C:\Users\gaele\OneDrive\Desktop\Hackaton\transacciones.json",
]

CONVERSATIONS_PATH = r"C:\Users\gaele\OneDrive\Desktop\Hackaton\conversaciones.json"

USER_ID_PATTERN = re.compile(r"^USR-\d{5}$", re.IGNORECASE)

# =========================
# KEYWORDS DE ASESORÍA
# =========================
KEYWORDS_ASESORIA = [
    "crecer", "invertir", "ahorrar", "negocio", "gastos", "presupuesto",
    "administrar", "planificar", "estrategia", "ingreso", "deuda",
    "pagar", "capital", "rentable", "flujo de efectivo", "hacer crecer",
    "generar", "dinero", "mejorar", "finanzas", "ahorro", "inversion",
]

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


def load_conversations() -> pd.DataFrame:
    if not os.path.exists(CONVERSATIONS_PATH):
        return pd.DataFrame(columns=["user_id", "input", "output", "date", "conv_id"])

    df = to_df(load_json(CONVERSATIONS_PATH))
    df = normalize_columns(df)

    required_columns = ["user_id", "input", "output", "date", "conv_id"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    return df


CLIENTS_DF = load_all()
CONVERSATIONS_DF = load_conversations()

# =========================
# MEMORIA DE SESIONES
# =========================
sessions: dict[str, str] = {}
session_histories: dict[str, list[dict[str, str]]] = {}


def get_user(session_id: str) -> str | None:
    return sessions.get(session_id)


def save_user(session_id: str, user_id: str) -> None:
    sessions[session_id] = user_id


def is_valid_id(user_input: str) -> bool:
    return bool(USER_ID_PATTERN.fullmatch(str(user_input).strip().upper()))


def append_session_turn(session_id: str, role: str, message: str) -> None:
    history = session_histories.setdefault(session_id, [])
    history.append({"role": role, "message": message})
    session_histories[session_id] = history[-20:]


def get_session_history_text(session_id: str) -> str:
    history = session_histories.get(session_id, [])
    if not history:
        return "Sin contexto previo en esta sesion."

    lines = [f"{item.get('role', 'desconocido')}: {item.get('message', '')}" for item in history]
    return "\n".join(lines)


def get_user_conversation_history(user_id: str, limit: int = 8) -> list[dict[str, str]]:
    if CONVERSATIONS_DF.empty:
        return []

    rows = CONVERSATIONS_DF[
        CONVERSATIONS_DF["user_id"].astype(str).str.upper() == str(user_id).upper()
    ].copy()

    if rows.empty:
        return []

    rows["_date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.sort_values(by=["_date"], kind="stable")

    history: list[dict[str, str]] = []
    for record in rows.tail(limit).to_dict(orient="records"):
        history.append(
            {
                "input": str(record.get("input") or ""),
                "output": str(record.get("output") or ""),
                "date": str(record.get("date") or ""),
                "conv_id": str(record.get("conv_id") or ""),
            }
        )

    return history


def seed_session_history_with_user_context(session_id: str, user_id: str) -> list[dict[str, str]]:
    history = get_user_conversation_history(user_id, limit=6)
    session_histories[session_id] = []

    for item in history:
        if item["input"]:
            append_session_turn(session_id, "usuario_anterior", item["input"])
        if item["output"]:
            append_session_turn(session_id, "asistente_anterior", item["output"])

    return history


def build_personalized_welcome(user_id: str, history: list[dict[str, str]]) -> str:
    if not history:
        return f"ID registrada ({user_id}). No encontre conversaciones previas. Ahora dime en que puedo ayudarte."

    last_message = ""
    for item in reversed(history):
        if item.get("input"):
            last_message = item["input"]
            break

    if not last_message:
        return (
            f"ID registrada ({user_id}). Cargue tu contexto anterior para darte respuestas mas personalizadas. "
            "Dime como te ayudo hoy."
        )

    return (
        f"ID registrada ({user_id}). Ya revise tu contexto anterior. "
        f"Tu ultima consulta fue: '{last_message}'. "
        "Si quieres, continuamos desde ahi o hacemos una consulta nueva."
    )


def build_response(session_id: str, user_message: str, response_text: str) -> dict[str, str]:
    append_session_turn(session_id, "usuario", user_message)
    append_session_turn(session_id, "asistente", response_text)
    return {"respuesta": response_text}

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
        "dias_desde_ultimo_login",
        "nombre_producto",
        "tipo_producto",
        "producto_bancario",
    ]

    products = []
    for col in possible_columns:
        if col in rows.columns:
            values = rows[col].dropna().astype(str).tolist()
            products.extend(values)

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
        "dias desde ultimo login": record.get("dias_desde_ultimo_login"),
        "saldo": record.get("saldo"),
        "estado": record.get("estado"),
        "ciudad": record.get("ciudad"),
        "ingreso_mensual_mxn": record.get("ingreso_mensual_mxn"),
        "num_productos_activos": record.get("num_productos_activos"),
        "score_buro": record.get("score_buro"),
        "ocupacion": record.get("ocupacion"),
    }

    return summary


def get_available_fields() -> list[str]:
    return [col for col in CLIENTS_DF.columns if col != "__source_file__"]


def ask_llm_general(user_id: str, mensaje: str, user_rows: pd.DataFrame, chat_context: str) -> str:
    user_data = user_rows.to_dict(orient="records")

    prompt = f"""
Eres un asesor financiero bancario. Tienes acceso a los datos reales del usuario autenticado.

USUARIO AUTENTICADO: {user_id}
DATOS REALES DEL USUARIO: {user_data}
CONTEXTO RECIENTE DEL CHAT: {chat_context}
PREGUNTA: {mensaje}

INSTRUCCIONES:

Primero, clasifica internamente la pregunta en uno de estos dos tipos:

TIPO A — Consulta de datos bancarios:
Si la pregunta pide información específica del usuario (saldo, productos, movimientos, estado de cuenta,
historial, etc.), responde ÚNICAMENTE con base en los datos reales del JSON.
Si el dato no existe, dilo claramente. No inventes información.

TIPO B — Pregunta de asesoría financiera o de negocios:
Si la pregunta es sobre estrategias, consejos, cómo ahorrar, invertir, hacer crecer un negocio,
administrar gastos, planificación financiera, etc., responde como un asesor financiero experto.
Puedes usar el contexto de los datos del usuario (como su saldo, ingresos, ocupación) para
personalizar tu consejo, pero NO estás limitado a esos datos para responder.
Da consejos concretos, prácticos y accionables. No remitas al usuario a "consultar un experto"
ya que TÚ eres ese experto.

REGLAS GENERALES:
- Responde siempre en español.
- Sé conciso y claro.
- Si tienes datos del usuario relevantes para la pregunta, úsalos para personalizar la respuesta.
- Nunca digas "no tengo información" para preguntas de asesoría general — esas no requieren datos.
- Nunca inventes datos bancarios específicos del usuario.
- No menciones el tipo de pregunta ni la clasificación interna. Ve directo a la respuesta.
- Nunca escribas encabezados, títulos, prefijos como "Tipo A:", "Tipo B:", "Respuesta:" ni ninguna etiqueta antes de responder. Ve directo al contenido.
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
        history = seed_session_history_with_user_context(session_id, clean_user_id)
        return {"respuesta": build_personalized_welcome(clean_user_id, history)}

    user_rows = get_user_rows(user_id)
    if user_rows.empty:
        return {"respuesta": f"No se encontraron datos para el ID {user_id}."}

    mensaje_lower = mensaje.lower()
    es_asesoria = any(kw in mensaje_lower for kw in KEYWORDS_ASESORIA)

    try:
        # Si mezcla saldo + asesoría, mandar al LLM con el saldo en contexto
        solo_consulta_saldo = "saldo" in mensaje_lower and not es_asesoria

        if solo_consulta_saldo:
            saldo = get_user_balance(user_id)
            if saldo is None:
                return build_response(session_id, mensaje, "No encontre informacion de saldo para este usuario.")
            return build_response(session_id, mensaje, f"Tu saldo actual es {saldo} MXN.")

        if "estado de cuenta" in mensaje_lower or "estado cuenta" in mensaje_lower:
            summary = get_account_summary(user_id)
            if summary is None:
                return build_response(
                    session_id,
                    mensaje,
                    "No encontre informacion de estado de cuenta para este usuario.",
                )

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
            return build_response(session_id, mensaje, respuesta)

        if (
            "productos" in mensaje_lower
            or "producto bancario" in mensaje_lower
            or "mis productos" in mensaje_lower
        ):
            products = get_user_products(user_id)
            if not products:
                return build_response(
                    session_id,
                    mensaje,
                    "No encontre productos bancarios registrados para este usuario.",
                )
            return build_response(session_id, mensaje, "Tus productos bancarios son: " + ", ".join(products))

        if "que informacion" in mensaje_lower or "campos disponibles" in mensaje_lower:
            fields = get_available_fields()
            return build_response(session_id, mensaje, "Puedo consultar estos campos: " + ", ".join(fields))

        # Para asesoría (con o sin saldo mencionado), enriquecer con saldo si existe
        chat_context = get_session_history_text(session_id)

        if es_asesoria:
            saldo = get_user_balance(user_id)
            if saldo:
                mensaje_enriquecido = f"{mensaje}\n\n[Contexto adicional: el saldo actual del usuario es {saldo} MXN]"
            else:
                mensaje_enriquecido = mensaje
            respuesta = ask_llm_general(user_id, mensaje_enriquecido, user_rows, chat_context)
        else:
            respuesta = ask_llm_general(user_id, mensaje, user_rows, chat_context)

        return build_response(session_id, mensaje, respuesta)

    except Exception as e:
        return {"respuesta": f"Error interno del servidor: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "ok"}