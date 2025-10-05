
# geolocalizzazione.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import traceback
from sqlalchemy import create_engine
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import re
import unicodedata
# --- Config Globali/DB ---
DB_FILE = "dati_georoute.db"
CONN_STRING = f"sqlite:///{DB_FILE}"
GEOCODE_DELAY = 1  # secondi
TABLE_PV = "PuntiVendita"
TABLE_PA = "PoliClienti"
TABLE_CONC = "Concorrenti"

#helpers per normalizzazione e riconoscimento tabelle concorrenti diverse
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _canon(col: str) -> str:
    # canonizza un nome colonna: minuscolo, senza accenti, alfanumerico+spazi ridotti
    c = _strip_accents(str(col)).lower()
    c = re.sub(r'[^a-z0-9 ]+', ' ', c)
    c = re.sub(r'\s+', ' ', c).strip()
    return c

def _pick_col(df, candidates) -> str | None:
    # prova a trovare nel df una colonna che matcha uno dei candidati (canonizzati)
    cols = { _canon(c): c for c in df.columns }
    for cand in candidates:
        if cand in cols:         # cand già canonico
            return cols[cand]
        # prova anche startswith/contains su canonici
        for can_key, orig in cols.items():
            if can_key == cand or can_key.startswith(cand) or cand in can_key:
                return orig
    return None

def normalize_concorrenti_df(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converte un DataFrame concorrenti (con header 'strani') nel formato standard:
    ['STRUTTURA','TIPOLOGIA','Indirizzo','CAP','Comune','Provincia','Regione','Latitudine','Longitudine']
    Restituisce (df_norm, mapping_usato)
    """
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]

    # Candidati canonici (lasciare in minuscolo e "canonici" con _canon)
    C = {
        'STRUTTURA': ['struttura','denominazione','ragione sociale','insegna','nome','negozio','esercizio'],
        'TIPOLOGIA': ['tipologia','categoria','tipo','segmento','tipo rete'],
        'Indirizzo': ['indirizzo','via','indirizzo completo','address','indirizzo 1','indirizzo1','indirizzo e numero civico'],
        'CAP': ['cap','zip','cap zip','postcode','cap/zip'],
        'Comune': ['comune','citta','città','localita','località','city','town'],
        'Provincia': ['provincia','prov','sigla provincia','pr'],
        'Regione': ['regione','reg','region','regione istat'],
        'Latitudine': ['latitudine','lat','latitude'],
        'Longitudine': ['longitudine','lon','long','lng','longitude'],
        # extra utili se presenti
        'Civico': ['civico','nr','numero','n','num'],
    }

    # Pre-costruisco indice canonico -> col originale
    canon2orig = { _canon(c): c for c in df.columns }

    used = {}
    out = {}

    def grab(key):
        if key in out: return out[key]
        col = _pick_col(df, C[key])
        if col is not None:
            used[key] = col
            out[key] = df[col]
        else:
            used[key] = None
            out[key] = pd.Series([None]*len(df))
        return out[key]

    # Campi principali
    struttura  = grab('STRUTTURA')
    tipologia  = grab('TIPOLOGIA')
    indirizzo  = grab('Indirizzo')
    civico     = grab('Civico')
    cap        = grab('CAP')
    comune     = grab('Comune')
    provincia  = grab('Provincia')
    regione    = grab('Regione')
    lat        = grab('Latitudine')
    lon        = grab('Longitudine')

    # Se ho 'Via' + 'Civico' ma manca 'Indirizzo', concateno
    if used['Indirizzo'] is None and used['Civico'] is not None:
        indirizzo = (grab('Indirizzo').fillna('') + ' ' + civico.fillna('')).str.strip()

    # CAP: se manca prova ad estrarlo dall'indirizzo
    cap = cap.astype(str).str.extract(r'(\d{5})', expand=False)
    if cap.isna().all():
        cap_from_addr = indirizzo.astype(str).str.extract(r'(\d{5})', expand=False)
        cap = cap_from_addr.where(cap.notna(), cap_from_addr)

    # Provincia: normalizza eventuale sigla
    provincia = provincia.astype(str).str.strip().str.upper().str.replace(r'[^A-Z]', '', regex=True)

    # Regione: normalizza (EMILIA-ROMAGNA -> EMILIA ROMAGNA)
    regione = regione.astype(str).str.strip().str.upper().str.replace('-', ' ', regex=False)
    regione = regione.str.replace(r'\s+', ' ', regex=True)

    # Lat/Lon numerici
    try:
        lat = pd.to_numeric(lat, errors='coerce')
    except Exception:
        lat = pd.Series([np.nan]*len(df))
    try:
        lon = pd.to_numeric(lon, errors='coerce')
    except Exception:
        lon = pd.Series([np.nan]*len(df))

    df_norm = pd.DataFrame({
        'STRUTTURA': struttura.fillna('').astype(str).str.strip(),
        'TIPOLOGIA': tipologia.fillna('N/A').astype(str).str.strip(),
        'Indirizzo': indirizzo.fillna('').astype(str).str.strip(),
        'CAP': cap.fillna('').astype(str).str.strip(),
        'Comune': comune.fillna('').astype(str).str.strip(),
        'Provincia': provincia.fillna('').astype(str).str.strip(),
        'Regione': regione.fillna('').astype(str).str.strip(),
        'Latitudine': lat,
        'Longitudine': lon,
    })

    # Drop righe troppo vuote (senza struttura e indirizzo)
    df_norm = df_norm[~(df_norm['STRUTTURA'].eq('') & df_norm['Indirizzo'].eq(''))].copy()

    # Dedup ragionevole
    df_norm.drop_duplicates(subset=['STRUTTURA','Comune','Indirizzo'], inplace=True)

    return df_norm, used
def import_concorrenti_files(files, engine, mode="append"):
    """
    Legge più file concorrenti (xlsx/xls), normalizza le colonne e salva su SQLite.
    mode: "append" per accodare, "replace" per sostituire l'intera tabella.
    """
    if not files:
        st.warning("Seleziona almeno un file Concorrenti.")
        return

    all_rows = []
    logs = []

    for f in files:
        try:
            raw = pd.read_excel(f)
        except Exception as e:
            st.error(f"Errore apertura '{getattr(f,'name',str(f))}': {e}")
            continue

        df_norm, used = normalize_concorrenti_df(raw)
        df_norm = _normalize_region_values(df_norm)
        all_rows.append(df_norm)

        # Log mapping per file
        logs.append({
            'file': getattr(f,'name',str(f)),
            'STRUTTURA': used.get('STRUTTURA'),
            'TIPOLOGIA': used.get('TIPOLOGIA'),
            'Indirizzo': used.get('Indirizzo'),
            'CAP': used.get('CAP'),
            'Comune': used.get('Comune'),
            'Provincia': used.get('Provincia'),
            'Regione': used.get('Regione'),
            'Latitudine': used.get('Latitudine'),
            'Longitudine': used.get('Longitudine'),
            'righe_importate': len(df_norm)
        })

    if not all_rows:
        st.error("Nessun file valido importato.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # Se append, unisci con tabella esistente e dedup
    if mode == "append":
        try:
            existing = get_data_from_sqlite(TABLE_CONC, engine)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            df_all = pd.concat([existing, df_all], ignore_index=True)
            df_all.drop_duplicates(subset=['STRUTTURA','Comune','Indirizzo'], inplace=True)

    if load_data_to_sqlite(df_all, TABLE_CONC, engine, mode="replace"):
        st.success(f"Concorrenti salvati ({len(df_all)} righe).")
        st.session_state.df_conc = get_data_from_sqlite(TABLE_CONC, engine)

    # Mostra log mapping
    if logs:
        with st.expander("Dettaglio mappatura colonne per file"):
            st.dataframe(pd.DataFrame(logs), use_container_width=True)


#----fine helpers




# Geocoders (istanziati una volta)
try:
    geolocator_nominatim = Nominatim(user_agent="georoute_app_nominatim")
except Exception as e:
    st.error(f"Nominatim init error: {e}")
    geolocator_nominatim = None

try:
    geolocator_arcgis = ArcGIS(user_agent="georoute_app_arcgis")
except Exception as e:
    st.error(f"ArcGIS init error: {e}")
    geolocator_arcgis = None

@st.cache_resource(show_spinner=False)
def get_db_engine():
    return create_engine(CONN_STRING)

def get_data_from_sqlite(table_name, engine):
    try:
        return pd.read_sql_table(table_name, engine)
    except Exception as e:
        if "no such table" in str(e).lower():
            return pd.DataFrame()
        st.error(f"Errore lettura tabella {table_name}: {e}")
        return pd.DataFrame()

def load_data_to_sqlite(df, table_name, engine, mode="replace"):
    """
    mode: "replace" (default) oppure "append"
    """
    try:
        # Forza tipi utili
        if table_name == TABLE_PV and 'ID' in df.columns:
            df['ID'] = df['ID'].fillna('').astype(str)
        for c in ('Latitudine','Longitudine'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.to_sql(table_name, engine, if_exists=mode, index=False)
        return True
    except Exception as e:
        st.error(f"Errore salvataggio in {table_name}: {e}")
        traceback.print_exc()
        return False

# --- Normalizzazioni ---
def _normalize_region_values(df):
    if 'Regione' in df.columns:
        df['Regione'] = (
            df['Regione'].astype(str).str.strip().str.upper()
            .str.replace('-', ' ', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
        )
    return df

def validate_and_clean_df(df, logical_name):
    df.columns = df.columns.astype(str)
    # Fix comuni
    if logical_name == TABLE_PV:
        if 'Povincia' in df.columns and 'Provincia' not in df.columns:
            df.rename(columns={'Povincia':'Provincia'}, inplace=True)
        if 'ID' in df.columns:
            df['ID'] = df['ID'].fillna('').astype(str)
        if 'Tipo Rete' not in df.columns:
            df['Tipo Rete'] = 'Sconosciuta'
    elif logical_name == TABLE_CONC:
        # i file concorrenti spesso in maiuscolo
        df.columns = df.columns.str.upper()
        rename_map = {
            'REGIONE':'Regione','INDIRIZZO':'Indirizzo','CAP':'CAP',
            'COMUNE':'Comune','PROV':'Provincia','PROVINCIA':'Provincia'
        }
        for k,v in rename_map.items():
            if k in df.columns: df.rename(columns={k:v}, inplace=True)
        if 'STRUTTURA' not in df.columns:
            df['STRUTTURA'] = 'N/A'
        if 'TIPOLOGIA' not in df.columns:
            df['TIPOLOGIA'] = 'N/A'

    # Colonne richieste minime
    required = ['Regione','Indirizzo','CAP','Comune','Provincia']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Mancano colonne obbligatorie: {', '.join(missing)}")
        return None

    # CAP come stringa senza .0
    df['CAP'] = df['CAP'].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan','', regex=False).str.strip()

    # trim principali
    for c in required:
        df[c] = df[c].fillna('').astype(str).str.strip()

    # lat/lon se mancanti
    if 'Latitudine' not in df.columns: df['Latitudine'] = np.nan
    if 'Longitudine' not in df.columns: df['Longitudine'] = np.nan

    df = _normalize_region_values(df)
    return df

# --- Geocoding ---
def attempt_geocode(geolocator, address, source_name, ref_text):
    try:
        import time as _t
        _t.sleep(GEOCODE_DELAY)
        loc = geolocator.geocode(address, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, f"{ref_text} ({source_name})"
        return None, None, None
    except Exception:
        return None, None, None

def geocode_address(row):
    if pd.notna(row['Latitudine']) and pd.notna(row['Longitudine']):
        return row['Latitudine'], row['Longitudine'], f"Già geocodificato: {row['Indirizzo']}, {row['Comune']}"
    full_addr = f"{row['Indirizzo']}, {row['CAP']} {row['Comune']}, {row['Provincia']}, Italia"
    simple_addr = f"{row['Comune']}, {row['Provincia']}, Italia"

    if geolocator_nominatim:
        lat, lon, s = attempt_geocode(geolocator_nominatim, full_addr, "Nominatim Completo", full_addr)
        if lat is not None: return lat, lon, s
        lat, lon, s = attempt_geocode(geolocator_nominatim, simple_addr, "Nominatim Semplificato", full_addr)
        if lat is not None: return lat, lon, s

    if geolocator_arcgis:
        lat, lon, s = attempt_geocode(geolocator_arcgis, full_addr, "ArcGIS", full_addr)
        if lat is not None: return lat, lon, s

    return None, None, full_addr + " (FALLITO)"

def perform_geocoding(df, table_name, engine):
    to_geocode = df[(df['Latitudine'].isna()) | (df['Longitudine'].isna())].copy()
    if to_geocode.empty:
        st.success(f"Tutti i record in '{table_name}' sono già geocodificati.")
        return df

    progress = st.progress(0, text="Geocodifica in corso...")
    log_container = st.empty()
    live = []
    tot = len(to_geocode)

    for i,(idx,row) in enumerate(to_geocode.iterrows(), start=1):
        lat, lon, msg = geocode_address(row)
        df.loc[idx,'Latitudine'] = lat
        df.loc[idx,'Longitudine'] = lon
        live.append({'#': i, 'Stato': 'OK' if lat is not None else 'KO', 'Indirizzo': msg})
        with log_container.container():
            st.dataframe(pd.DataFrame(live).tail(10), use_container_width=True, hide_index=True)
        progress.progress(i/tot, text=f"Geocodifica {i}/{tot}")

    progress.empty()
    log_container.empty()

    if not load_data_to_sqlite(df, table_name, engine):
        st.error("Salvataggio su SQLite fallito.")
    return df

# --- UI Geolocalizzazione ---
def render_geocoding_ui(engine):
    # ===== Stili "card" (solo header colorati) =====
    _CARD_CSS = """
    <style>
      .block-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
        max-width: 100% !important;
      }
      .block-container h1, .block-container h2, .block-container h3 {
        margin-top: 0.3rem !important;
      }
      .card-header { 
        padding: 10px 14px; 
        border-radius: 10px; 
        color: white; 
        font-weight: 600; 
        margin-bottom: 10px;
      }
      .card-pa   { background: linear-gradient(90deg,#2b6cb0,#3182ce); }
      .card-pv   { background: linear-gradient(90deg,#2f855a,#38a169); }
      .card-conc { background: linear-gradient(90deg,#c53030,#e53e3e); }
      .card-box  { padding: 12px; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; background: #ffffffAA; }
    </style>
    """
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    st.subheader("Caricamento & Geocodifica")

    # ===== Tre box affiancati: PA / PV / CONC =====
    col_pa, col_pv, col_conc = st.columns(3, gap="large")

    # --- Poli Clienti ---
    with col_pa:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-header card-pa">PoliAmbulatori</div>', unsafe_allow_html=True)
        up_pa = st.file_uploader("Seleziona file", type=["xlsx","xls"], key="up_pa_page", label_visibility="collapsed")
        st.caption("Drag and drop file here  •  Limit 200MB per file  •  XLSX, XLS")
        if up_pa is not None:
            try:
                new_df = pd.read_excel(up_pa)
                dfp = validate_and_clean_df(new_df, TABLE_PA)
                if dfp is not None and load_data_to_sqlite(dfp, TABLE_PA, engine, mode="replace"):
                    st.session_state.df_pa = get_data_from_sqlite(TABLE_PA, engine)
                    st.success("PoliAmbulatori salvati.")
                    st.rerun()
            except Exception as e:
                st.error(f"Errore import PA: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Punti Vendita ---
    with col_pv:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-header card-pv">Punti di Interesse</div>', unsafe_allow_html=True)
        up_pv = st.file_uploader("Seleziona file", type=["xlsx","xls"], key="up_pv_page", label_visibility="collapsed")
        st.caption("Drag and drop file here  •  Limit 200MB per file  •  XLSX, XLS")
        if up_pv is not None:
            try:
                new_df = pd.read_excel(up_pv)
                dfp = validate_and_clean_df(new_df, TABLE_PV)
                if dfp is not None and load_data_to_sqlite(dfp, TABLE_PV, engine, mode="replace"):
                    # Se manca ID, lo creo coerente
                    if 'ID' not in st.session_state.df_pv.columns:
                        st.session_state.df_pv['ID'] = st.session_state.df_pv.index.astype(str)
                    st.session_state.df_pv = get_data_from_sqlite(TABLE_PV, engine)
                    st.success("Punti di Interesse salvati.")
                    st.rerun()
            except Exception as e:
                st.error(f"Errore import PV: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Concorrenti (multi-file) ---
    with col_conc:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-header card-conc">Concorrenti</div>', unsafe_allow_html=True)
        up_conc_multi = st.file_uploader(
            "Seleziona uno o più file",
            type=["xlsx","xls"],
            key="up_conc_multi_page",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.caption("Drag and drop file here  •  Limit 200MB per file  •  XLSX, XLS")
        mode_conc = st.radio(
            "Salvataggio",
            ["Sostituisci tabella", "Aggiungi alla tabella esistente"],
            index=1,
            horizontal=True
        )
        if st.button("Importa Concorrenti", use_container_width=True):
            import_concorrenti_files(
                up_conc_multi, engine,
                mode="append" if mode_conc == "Aggiungi alla tabella esistente" else "replace"
            )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== Pulsanti Geocodifica (senza preview) =====
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Geocodifica Poli Clienti", key="geo_pa_btn", use_container_width=True):
            df_pa = st.session_state.get('df_pa', pd.DataFrame())
            if df_pa.empty:
                st.warning("Nessun dato Poli Clienti da geocodificare.")
            else:
                st.session_state.df_pa = perform_geocoding(df_pa.copy(), TABLE_PA, engine)
                st.rerun()
    with c2:
        if st.button("Geocodifica Locazioni", key="geo_pv_btn", use_container_width=True):
            df_pv = st.session_state.get('df_pv', pd.DataFrame())
            if df_pv.empty:
                st.warning("Nessun dato da geocodificare.")
            else:
                st.session_state.df_pv = perform_geocoding(df_pv.copy(), TABLE_PV, engine)
                st.rerun()
    with c3:
        if st.button("Geocodifica Concorrenti", key="geo_conc_btn", use_container_width=True):
            df_conc = st.session_state.get('df_conc', pd.DataFrame())
            if df_conc.empty:
                st.warning("Nessun dato da geocodificare.")
            else:
                st.session_state.df_conc = perform_geocoding(df_conc.copy(), TABLE_CONC, engine)
                st.rerun()
