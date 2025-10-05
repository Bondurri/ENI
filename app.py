import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
from sqlalchemy import text

from geolocalizzazione import (
    get_db_engine,
    get_data_from_sqlite,
    render_geocoding_ui,
    TABLE_PV, TABLE_PA, TABLE_CONC
)
from interaction import render_interaction_ui
from foresee import render_foresee_ui

# ==========================================
# 🔐 LOGIN E LOG ACCESSI
# ==========================================

engine = get_db_engine()

# Creazione tabelle Users e Log se non esistono
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            access_time TEXT,
            ip TEXT
        )
    """))

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, hash_value: str) -> bool:
    return hash_password(password) == hash_value

def log_access(username, ip="local"):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO Log (username, access_time, ip) VALUES (:u, :t, :i)"),
            {"u": username, "t": datetime.now().isoformat(timespec='seconds'), "i": ip}
        )

def check_user(username, password):
    with engine.begin() as conn:
        res = conn.execute(text("SELECT password_hash FROM Users WHERE username=:u"), {"u": username}).fetchone()
        if res and verify_password(password, res[0]):
            return True
    return False

# ==========================================
# 🔑 INTERFACCIA LOGIN
# ==========================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(layout="centered", page_title="Login – GeoRoute Optimizer")
    st.title("🔐 Accesso riservato")
    user = st.text_input("Utente")
    pw = st.text_input("Password", type="password")

    if st.button("Accedi"):
        if check_user(user, pw):
            st.session_state.authenticated = True
            st.session_state.username = user
            log_access(user)
            st.success(f"Benvenuto {user}")
            st.rerun()
        else:
            st.error("Credenziali non valide")

    st.stop()

# ==========================================
# 🌍 APP PRINCIPALE DOPO LOGIN
# ==========================================

st.set_page_config(layout="wide", page_title="GeoRoute Optimizer Italia")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        min-width: 255px;
        max-width: 255px;
        background-color: #fffde7;
        color: black;
        text-align: center;
    }

    [data-testid="stSidebar"] img {
        margin: 6px auto 20px auto;
        display: block;
        max-width: 50% !important;
    }

    .sidebar-title {
        font-size: 10pt;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .sidebar-btn {
        display: block;
        width: 100%;
        background-color: #fde047;
        border: 1.5px solid #9ca3af;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 6px 0;
        font-size: 14px;
        font-weight: 600;
        color: black;
        text-align: center;
        text-decoration: none;
    }
    .sidebar-btn:hover { background-color: #facc15; }
    .sidebar-btn.active { background-color: #ca8a04; color: white; }

    .block-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
        max-width: 100% !important;
    }
    .block-container h1, .block-container h2, .block-container h3 {
        margin-top: 0.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Titolo sopra il logo ---
st.sidebar.markdown("<div class='sidebar-title'>ENI Energia e Salute</div>", unsafe_allow_html=True)

# --- Logo ---
st.sidebar.image("logo.png", use_container_width=True)

# --- Sidebar navigation custom ---
pages = {
    "Help": "📖 Help",
    "GeoLocalizzazione": "📍 GeoLocalizzazione",
    "Interazione": "🧭 Interazione",
    "Foresee": "🔮 Foresee"
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "Help"

for key, label in pages.items():
    active = "active" if st.session_state.current_page == key else ""
    st.sidebar.markdown(
        f"<a class='sidebar-btn {active}' href='?page={key}'>{label}</a>",
        unsafe_allow_html=True
    )

# --- Query params ---
query_params = st.query_params
if "page" in query_params:
    st.session_state.current_page = query_params["page"]

page = st.session_state.current_page

# --- Sidebar info utente ---
st.sidebar.markdown(f"👤 **Utente:** {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Inizializzazione dati ---
def initialize_df_state(table_name, required_cols):
    df_init = get_data_from_sqlite(table_name, engine)
    if 'Latitudine' not in df_init.columns: df_init['Latitudine'] = np.nan
    if 'Longitudine' not in df_init.columns: df_init['Longitudine'] = np.nan
    for col in required_cols:
        if col not in df_init.columns:
            df_init[col] = None
    return df_init

if 'df_pa' not in st.session_state:
    st.session_state.df_pa = initialize_df_state(
        TABLE_PA, ['Ragione sociale fornitore','Indirizzo','CAP','Comune','Provincia','Regione']
    )

if 'df_pv' not in st.session_state:
    df_pv_init = initialize_df_state(
        TABLE_PV, ['Regione','Indirizzo','CAP','Comune','Provincia','Tipo Rete']
    )
    if 'ID' not in df_pv_init.columns:
        df_pv_init['ID'] = df_pv_init.index.astype(str)
    st.session_state.df_pv = df_pv_init

if 'df_conc' not in st.session_state:
    st.session_state.df_conc = initialize_df_state(
        TABLE_CONC, ['STRUTTURA','TIPOLOGIA','Indirizzo','CAP','Comune','Provincia','Regione']
    )

df_pa = st.session_state.df_pa
df_pv = st.session_state.df_pv
df_conc = st.session_state.df_conc

# --- Render pagine ---
if page == "Help":
    try:
        html = Path("help_benzinai.html").read_text(encoding="utf-8")
        st.components.v1.html(html, height=1800, scrolling=True)
    except FileNotFoundError:
        st.error("File help_benzinai.html non trovato. Caricalo nella cartella dell'app.")

elif page == "GeoLocalizzazione":
    render_geocoding_ui(engine)

elif page == "Interazione":
    render_interaction_ui(engine)

elif page == "Foresee":
    render_foresee_ui()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:12px;color:#111;'>Ver. 1.0<br><strong>Luigi Bondurri</strong></div>",
    unsafe_allow_html=True
)
