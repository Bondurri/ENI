# interaction.py
# -*- coding: utf-8 -*-
"""
Streamlit UI: Interazione – Mappe, Vicinanza e Routing
- Tab 1: Percorsi & Mappa
- Tab 2: Ricerca PoliAmbulatorio
Fix principali:
- Colonna nome Polo normalizzata a "Ragione sociale Fornitore" (auto-rename)
- Ricerca su Regione + testo (nome/via/città)
- Raggiungibili per tempo (ORS matrix 1→N) con riepilogo 15/30/45/60
- Best routing Centro→Cliente selezionato con istruzioni IT (language='it')
- Mappa che resta stabile usando st.session_state
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import FastMarkerCluster

# ------------------------------
# Costanti / Config
# ------------------------------
PA_COL = "Ragione sociale Fornitore"   # nome canonico per il Polo/Cliente
ORS_LANGUAGE = "it"                    # istruzioni turn-by-turn in italiano

# ------------------------------
# Helper su colonne / dati
# ------------------------------
def ensure_pa_name_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se nel DF non esiste PA_COL, rinomina automaticamente
    la prima colonna candidata tra i sinonimi -> PA_COL.
    """
    if df is None or df.empty:
        return df
    if PA_COL in df.columns:
        return df

    candidates = [
        "Ragione sociale Fornitore",
        "Ragione sociale fornitore",
        "Ragione sociale",
        "Ragione Sociale",
        "Denominazione",
        "Insegna",
        "Nome",
        "Fornitore",
    ]
    lc = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = lc.get(cand.lower().strip())
        if key:
            return df.rename(columns={key: PA_COL})
    return df  # nessun match: lascio com’è


def _normalize_region_values(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and 'Regione' in df.columns:
        df['Regione'] = (
            df['Regione'].astype(str).str.strip().str.upper()
            .str.replace('-', ' ', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
        )
    return df


@st.cache_data(show_spinner=False)
def get_regions(df_pa: pd.DataFrame, df_pv: pd.DataFrame, df_conc: pd.DataFrame) -> list[str]:
    def _pull(d, c): 
        return d[c].dropna().astype(str).tolist() if c in d.columns and not d.empty else []
    lst = _pull(df_pa, 'Regione') + _pull(df_pv, 'Regione') + _pull(df_conc, 'Regione')
    return sorted(list(set(lst)))


def get_ors_client():
    key = st.secrets.get("ORS_API_KEY")
    if not key:
        return None
    try:
        from openrouteservice import Client
        return Client(key=key)
    except Exception as e:
        st.error(f"Errore chiave ORS: {e}")
        return None


# ------------------------------
# Map rendering
# ------------------------------
def show_map(df_pa: pd.DataFrame, df_pv: pd.DataFrame, df_conc: pd.DataFrame, selected_region: str | None = None,
             route_geojson=None, filtered_pv_ids=None,
             selected_pa: pd.Series | None = None, show_pv: bool = False, draw_conc: bool = False):
    """
    - selected_pa: Series del Polo selezionato (o None)
    - show_pv: False -> non disegnare PV; True -> disegna solo quelli in filtered_pv_ids
    - filtered_pv_ids: lista di ID (string) dei PV da mostrare (se show_pv=True)
    - draw_conc: se True disegna anche concorrenti (default False per performance)
    """
    # Sottoselezione per regione
    if selected_region and selected_region != "Tutta Italia":
        df_pa_map = df_pa[df_pa['Regione'] == selected_region].copy()
        df_pv_map = df_pv[df_pv['Regione'] == selected_region].copy()
        df_conc_map = df_conc[df_conc['Regione'] == selected_region].copy()
    else:
        df_pa_map = df_pa.copy()
        df_pv_map = df_pv.copy()
        df_conc_map = df_conc.copy()

    # Filtri base
    for d in (df_pa_map, df_pv_map, df_conc_map):
        if not d.empty:
            d.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)

    # Mostra solo i PV calcolati
    if show_pv:
        if not df_pv_map.empty:
            if not filtered_pv_ids:
                df_pv_map = df_pv_map.iloc[0:0].copy()
            elif 'ID' in df_pv_map.columns:
                df_pv_map = df_pv_map[df_pv_map['ID'].astype(str).isin(filtered_pv_ids)].copy()
            else:
                df_pv_map = df_pv_map.iloc[0:0].copy()

    # Centro mappa
    if selected_pa is not None and pd.notna(selected_pa.get('Latitudine')) and pd.notna(selected_pa.get('Longitudine')):
        center_lat = float(selected_pa['Latitudine']); center_lon = float(selected_pa['Longitudine'])
    else:
        if not df_pa_map.empty:
            center_lat = float(df_pa_map['Latitudine'].mean())
            center_lon = float(df_pa_map['Longitudine'].mean())
        else:
            center_lat, center_lon = 41.8719, 12.5674  # centro Italia

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # Polo selezionato
    if selected_pa is not None and pd.notna(selected_pa.get('Latitudine')) and pd.notna(selected_pa.get('Longitudine')):
        pa_name = selected_pa.get(PA_COL, "N/A")
        popup = f"<b>Punti interesse:</b> {pa_name}<br>{selected_pa.get('Indirizzo','')} - {selected_pa.get('Comune','')}"
        folium.CircleMarker(
            location=[selected_pa['Latitudine'], selected_pa['Longitudine']],
            radius=10, color='blue', fill=True, fill_color='blue', fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=320)
        ).add_to(m)

    # PV (dopo calcolo)
    if show_pv and not df_pv_map.empty:
        pts = df_pv_map[['Latitudine','Longitudine']].to_numpy().tolist()
        if len(pts) > 800:
            FastMarkerCluster(pts).add_to(m)
        else:
            for _, row in df_pv_map.iterrows():
                popup = f"<b>Punto Vendita:</b> {row.get('Regione','')} - {row.get('Provincia','')}<br>{row.get('Indirizzo','')}"
                folium.CircleMarker(
                    location=[row['Latitudine'], row['Longitudine']],
                    radius=6, color='green', fill=True, fill_color='green', fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=320)
                ).add_to(m)

    # Concorrenti (opzionale)
    if draw_conc and not df_conc_map.empty:
        for _, row in df_conc_map.iterrows():
            popup = f"<b>Concorrente:</b> {row.get('STRUTTURA','')}<br>{row.get('Indirizzo','')} - {row.get('Comune','')}"
            folium.CircleMarker(
                location=[row['Latitudine'], row['Longitudine']],
                radius=5, color='red', fill=True, fill_color='red', fill_opacity=0.7,
                popup=folium.Popup(popup, max_width=320)
            ).add_to(m)

    # Route
    if route_geojson:
        folium.GeoJson(
            route_geojson, name='Percorso', style_function=lambda x: {'color':'red','weight':5,'opacity':0.8}
        ).add_to(m)

    # Fit bounds
    bounds_pts = []
    if selected_pa is not None and pd.notna(selected_pa.get('Latitudine')) and pd.notna(selected_pa.get('Longitudine')):
        bounds_pts.append([float(selected_pa['Latitudine']), float(selected_pa['Longitudine'])])
    if show_pv and not df_pv_map.empty:
        bounds_pts.extend(df_pv_map[['Latitudine','Longitudine']].to_numpy().tolist())
    if len(bounds_pts) >= 1:
        m.fit_bounds(bounds_pts, padding=(20, 20))

    folium.LayerControl().add_to(m)
    st.components.v1.html(m._repr_html_(), height=540)


# ------------------------------
# ORS utilities
# ------------------------------
def _distance_time_matrix(ors_client, start_coords: tuple[float, float], dest_coords: list[tuple[float, float]], max_batch=3000):
    """
    Ritorna (distances_m, durations_s) da start_coords verso dest_coords (1→N).
    """
    if ors_client is None or not dest_coords:
        return None, None

    # ORS vuole [lon, lat]
    coords_ors = [[start_coords[1], start_coords[0]]] + [[lon, lat] for lat, lon in dest_coords]
    all_dest_idx = list(range(1, len(coords_ors)))  # destinazioni

    distances_all, durations_all = [], []
    for i in range(0, len(all_dest_idx), max_batch):
        batch_dest = all_dest_idx[i:i+max_batch]
        try:
            matrix = ors_client.distance_matrix(
                locations=coords_ors,
                profile='driving-car',
                metrics=['duration', 'distance'],
                sources=[0],
                destinations=batch_dest
            )
            d = np.array(matrix.get('distances', [[np.nan]*len(batch_dest)])[0], dtype=float)
            t = np.array(matrix.get('durations', [[np.nan]*len(batch_dest)])[0], dtype=float)
        except Exception as e:
            st.error(f"Errore ORS batch {i}-{i+len(batch_dest)}: {e}")
            d = np.full(len(batch_dest), np.nan)
            t = np.full(len(batch_dest), np.nan)

        distances_all.append(d)
        durations_all.append(t)

    return np.concatenate(distances_all), np.concatenate(durations_all)


def _prefilter_by_haversine(start: tuple[float, float], dest_coords: list[tuple[float, float]], max_candidates=3000):
    """
    Prefiltro (linea d'aria) per tenere i K più vicini.
    Ritorna:
      - dest_coords_filtrate: [(lat,lon), ...]
      - idx_originali: indici nel dest_coords originale
      - distanze_km_filtrate: np.array
    """
    n = len(dest_coords)
    if n <= max_candidates:
        return dest_coords, np.arange(n), None

    R = 6371.0
    start_rad = np.radians(np.array(start, dtype=float))
    dest_rad = np.radians(np.array(dest_coords, dtype=float))
    d = np.sin((dest_rad[:,0]-start_rad[0])/2.0)**2 + \
        np.cos(start_rad[0])*np.cos(dest_rad[:,0]) * np.sin((dest_rad[:,1]-start_rad[1])/2.0)**2
    dist = 2*R*np.arcsin(np.sqrt(d))

    order = np.argsort(dist)[:max_candidates]
    dests_small = [dest_coords[i] for i in order]
    return dests_small, order, dist[order]


def compute_best_route_it(ors_client, start_lat: float, start_lon: float, dest_lat: float, dest_lon: float):
    """
    Calcola il percorso Centro -> Cliente con istruzioni IT.
    Ritorna: (route_geojson, dist_km, time_min, steps_df)
    """
    if ors_client is None:
        raise ValueError("ORS client mancante (impostare ORS_API_KEY in secrets).")

    coords_ors = [[start_lon, start_lat], [dest_lon, dest_lat]]
    route = ors_client.directions(
        coordinates=coords_ors,
        profile='driving-car',
        format='geojson',
        language=ORS_LANGUAGE,
        instructions=True
    )

    props = route['features'][0]['properties']
    summary = props.get('summary', {})
    dist_km = float(summary.get('distance', 0.0)) / 1000.0
    time_min = float(summary.get('duration', 0.0)) / 60.0

    steps_rows = []
    for seg in props.get('segments', []):
        for s in seg.get('steps', []):
            steps_rows.append({
                "Istruzione": s.get("instruction", ""),
                "Strada":     s.get("name", ""),
                "Km":         round(float(s.get('distance', 0.0)) / 1000.0, 2),
                "Min":        round(float(s.get('duration', 0.0)) / 60.0, 1),
            })
    steps_df = pd.DataFrame(steps_rows)
    return route, dist_km, time_min, steps_df


# ------------------------------
# UI principale
# ------------------------------
def render_interaction_ui(engine=None):
    st.markdown(
        """
        <style>
        .block-container {
            padding-left: 10px !important;
            padding-right: 10px !important;
            max-width: 100% !important;
        }
        .block-container h1, .block-container h2, .block-container h3 {
            margin-top: 0.3rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Interazione: Mappe, Vicinanza e Routing")

    # Dati da sessione + normalizzazione nome Polo e Regione
    df_pa = ensure_pa_name_col(st.session_state.get('df_pa', pd.DataFrame()).copy())
    df_pv = st.session_state.get('df_pv', pd.DataFrame()).copy()
    df_conc = st.session_state.get('df_conc', pd.DataFrame()).copy()

    df_pa = _normalize_region_values(df_pa)
    df_pv = _normalize_region_values(df_pv)
    df_conc = _normalize_region_values(df_conc)

    if df_pa.empty and df_pv.empty and df_conc.empty:
        st.info("Carica e geocodifica i dati nella tab 'Geolocalizzazione'.")
        return

    tab1, tab2 = st.tabs(["🗺️ Percorsi & Mappa (attuale)", "🧭 Ricerca PoliAmbulatorio"])

    # =========================
    # TAB 1
    # =========================
    with tab1:
        ors_client = get_ors_client()

        # Stato persistente
        ss = st.session_state
        for k, v in {
            'route_geojson': None,
            'filtered_pv_ids': [],
            'df_reach': pd.DataFrame(),
            'durations_min': None,
            'distances_km': None
        }.items():
            if k not in ss: ss[k] = v

        # dataset geocodificati
        df_pa_geo = df_pa.dropna(subset=['Latitudine','Longitudine']).copy()
        df_pv_geo = df_pv.dropna(subset=['Latitudine','Longitudine']).copy()
        df_conc_geo = df_conc.dropna(subset=['Latitudine','Longitudine']).copy()

        regions = get_regions(df_pa_geo, df_pv_geo, df_conc_geo)

        # Regione + Ricerca Polo sulla stessa riga
        col_region, col_search = st.columns([1, 1.2])
        with col_region:
            selected_region = st.selectbox(
                "Regione",
                ["Tutta Italia"] + regions,
                index=0,
                key="tab1_region"
            )
        with col_search:
            pa_search = st.text_input(
                "Cerca Polo (nome / via / città)",
                key="tab1_pa_search",
                placeholder=f"Digita per filtrare ({PA_COL}, indirizzo o comune)…"
            )

        # Filtro per regione
        if selected_region != "Tutta Italia":
            df_pa_reg = df_pa_geo[df_pa_geo['Regione'] == selected_region].copy()
            df_pv_reg = df_pv_geo[df_pv_geo['Regione'] == selected_region].copy()
        else:
            df_pa_reg = df_pa_geo.copy()
            df_pv_reg = df_pv_geo.copy()

        # Filtro testo Polo (nome/via/città)
        if pa_search and not df_pa_reg.empty:
            q = pa_search.strip()
            mask = (
                df_pa_reg[PA_COL].fillna('').str.contains(q, case=False, na=False) |
                df_pa_reg['Indirizzo'].fillna('').str.contains(q, case=False, na=False) |
                df_pa_reg['Comune'].fillna('').str.contains(q, case=False, na=False)
            )
            df_pa_reg = df_pa_reg[mask].copy()

        if df_pa_reg.empty:
            st.info(f"Nessun Punto di interesse geocodificato in **{selected_region}** con i criteri attuali.")
            show_map(df_pa_geo, df_pv_geo, df_conc_geo, selected_region, selected_pa=None, show_pv=False)
            st.stop()

        # Etichetta Polo: "Ragione — Via — Città"
        df_pa_reg = df_pa_reg.copy()
        df_pa_reg['__label'] = (
            df_pa_reg[PA_COL].fillna('') + ' — ' +
            df_pa_reg['Indirizzo'].fillna('') + ' — ' +
            df_pa_reg['Comune'].fillna('')
        )
        idx_options = list(range(len(df_pa_reg)))
        sel_idx = st.selectbox(
            "Seleziona un Punto di interesse",
            idx_options,
            format_func=lambda i: df_pa_reg['__label'].iloc[i],
            key="tab1_sel_pa_idx"
        )
        sel_pa = df_pa_reg.iloc[sel_idx]
        start = (float(sel_pa['Latitudine']), float(sel_pa['Longitudine']))

        # Dettagli Polo
        with st.expander("Dettagli Punto di Interesse selezionato", expanded=True):
            cols = st.columns([3, 2, 2, 2])
            with cols[0]:
                st.metric("Polo Cliente", sel_pa.get(PA_COL, "N/A"))
                st.write(f"{sel_pa.get('Indirizzo','')}")
                st.write(f"{sel_pa.get('CAP','')} {sel_pa.get('Comune','')} ({sel_pa.get('Provincia','')})")
            with cols[1]:
                st.metric("Regione", sel_pa.get("Regione", ""))
            with cols[2]:
                try: st.metric("Lat", f"{float(sel_pa.get('Latitudine')):.6f}")
                except: st.metric("Lat", "—")
            with cols[3]:
                try: st.metric("Lon", f"{float(sel_pa.get('Longitudine')):.6f}")
                except: st.metric("Lon", "—")

        # Limita PV alla stessa regione del Polo (default ON)
        limit_same_region = st.checkbox("Limita ai PV nella stessa Regione", value=True, key="tab1_same_region")
        if limit_same_region:
            df_pv_reg = df_pv_reg[df_pv_reg['Regione'] == sel_pa['Regione']].copy()

        # Slider tempo
        max_time_min = st.slider("Tempo massimo (minuti)", 15, 240, 60, 15, key="tab1_time")

        # Layout mappa + pannello destro
        summary_container = st.container()
        col_map, col_details = st.columns([7, 5])

        # Placeholder mappa
        map_placeholder = col_map.container()
        with map_placeholder:
            show_map(df_pa_geo, df_pv_geo, df_conc_geo, selected_region,
                     route_geojson=ss.get('route_geojson'),
                     filtered_pv_ids=ss.get('filtered_pv_ids', []),
                     selected_pa=sel_pa, show_pv=bool(ss.get('filtered_pv_ids')), draw_conc=False)

        # ---- LATO DESTRO: pulsante calcolo, riepilogo, tabella, selezione cliente, routing singolo ----
        with col_details:
            if ors_client is None:
                st.error("Inserisci la chiave ORS in .streamlit/secrets.toml per routing.")
            elif st.button("Calcola raggiungibili", type="primary", key="tab1_calc"):
                dests_all = list(zip(df_pv_reg['Latitudine'], df_pv_reg['Longitudine']))
                if not dests_all:
                    st.warning("Nessun Punto Vendita nella selezione corrente.")
                else:
                    # Prefiltro + matrice 1→N
                    dests, idx_map, _ = _prefilter_by_haversine(start, dests_all, max_candidates=3000)
                    d_m, t_s = _distance_time_matrix(ors_client, start, dests)
                    if d_m is not None:
                        durations_min = t_s / 60.0
                        distances_km = d_m / 1000.0

                        # Riepilogo 15/30/45/60
                        thresholds = [15, 30, 45, 60]
                        counts = [int((durations_min <= t).sum()) for t in thresholds]
                        summary_df = pd.DataFrame({"Soglia (min)": thresholds, "Clienti raggiungibili": counts})
                        with summary_container:
                            st.subheader("Riepilogo raggiungibili dal Polo")
                            st.dataframe(summary_df, use_container_width=False, hide_index=True)

                        # Entro slider
                        reachable_idx = np.where(durations_min <= max_time_min)[0]
                        if reachable_idx.size == 0:
                            st.error("Nessun Punto Vendita raggiungibile entro il tempo selezionato.")
                            ss['route_geojson'] = None
                            ss['filtered_pv_ids'] = []
                            ss['df_reach'] = pd.DataFrame()
                            ss['durations_min'] = None
                            ss['distances_km'] = None
                        else:
                            df_reach = df_pv_reg.iloc[idx_map[reachable_idx]].copy()
                            df_reach['Distanza_Km'] = distances_km[reachable_idx]
                            df_reach['Tempo_min'] = durations_min[reachable_idx]
                            df_reach.sort_values(by=['Distanza_Km','Tempo_min'], inplace=True, ignore_index=True)

                            ss['df_reach'] = df_reach
                            ss['durations_min'] = durations_min[reachable_idx]
                            ss['distances_km'] = distances_km[reachable_idx]
                            ss['filtered_pv_ids'] = df_reach.get('ID', df_reach.index.astype(str)).astype(str).tolist()

                            st.subheader(f"Raggiungibili in ≤ {max_time_min} min")
                            cols = [c for c in ['Regione','Provincia','Comune','Indirizzo','Tipo Rete','Distanza_Km','Tempo_min'] if c in df_reach.columns]
                            st.dataframe(df_reach[cols], use_container_width=True)

            # Se esistono raggiungibili, selezione cliente e routing singolo
            if not ss.get('df_reach', pd.DataFrame()).empty:
                df_reach = ss['df_reach']
                labels = (df_reach['Comune'].fillna('') + ' – ' + df_reach['Indirizzo'].fillna('')).tolist()
                sel_cli_idx = st.selectbox(
                    "Seleziona un cliente (per il best routing)",
                    list(range(len(df_reach))),
                    key="tab1_sel_result_idx",
                    format_func=lambda i: labels[i] if i < len(labels) else str(i)
                )

                # Calcola route solo centro -> cliente selezionato (2 waypoints) con istruzioni in ITA
                if ors_client is not None:
                    dest_lat = float(df_reach.iloc[sel_cli_idx]['Latitudine'])
                    dest_lon = float(df_reach.iloc[sel_cli_idx]['Longitudine'])
                    try:
                        route, dist_km, time_min, steps_df = compute_best_route_it(
                            ors_client, start[0], start[1], dest_lat, dest_lon
                        )
                        ss['route_geojson'] = route
                        # mostra solo quel PV sulla mappa (se abbiamo ID)
                        if 'ID' in df_reach.columns:
                            ss['filtered_pv_ids'] = [str(df_reach.iloc[sel_cli_idx]['ID'])]
                        else:
                            ss['filtered_pv_ids'] = []
                    except Exception as e:
                        st.warning(f"Percorso non disponibile: {e}")
                        ss['route_geojson'] = None
                        steps_df = pd.DataFrame()
                        dist_km = time_min = 0.0

                    # Dettaglio routing (best route)
                    if ss.get('route_geojson'):
                        st.subheader("Best routing selezionato")
                        cA, cB = st.columns(2)
                        with cA: st.metric("Distanza", f"{dist_km:.1f} km")
                        with cB: st.metric("Tempo", f"{time_min:.0f} min")

                        if not steps_df.empty:
                            with st.expander("Istruzioni turn-by-turn (italiano)"):
                                st.dataframe(steps_df, use_container_width=True, hide_index=True)

                # ridisegna la mappa con il route corrente
                with map_placeholder:
                    show_map(df_pa_geo, df_pv_geo, df_conc_geo, selected_region,
                             route_geojson=ss.get('route_geojson'),
                             filtered_pv_ids=ss.get('filtered_pv_ids', []),
                             selected_pa=sel_pa, show_pv=bool(ss.get('filtered_pv_ids')), draw_conc=False)

    # =========================
    # TAB 2: Ricerca Centro + filtri tempo/distanza (robust)
    # =========================
    with tab2:
        ors_client = get_ors_client()

        df_pa_geo = df_pa.dropna(subset=['Latitudine','Longitudine']).copy()
        df_pv_geo = df_pv.dropna(subset=['Latitudine','Longitudine']).copy()
        df_conc_geo = df_conc.dropna(subset=['Latitudine','Longitudine']).copy()

        regions = get_regions(df_pa_geo, df_pv_geo, df_conc_geo)

        # Filtro ricerca centro + regione (stessa riga)
        c1, c2 = st.columns([3,2])
        with c1:
            query = st.text_input("Cerca Polo Cliente (nome / via / città)", "", key="tab2_query",
                                  placeholder=f"Digita parte di {PA_COL}, indirizzo o città…")
        with c2:
            region_filter = st.selectbox("Filtro Regione (opzionale)", ["(tutte)"] + regions, index=0, key="tab2_region")

        # Applica filtri al dataset Poli
        df_search = df_pa_geo.copy()
        if query.strip():
            q = query.strip()
            mask = (
                df_search[PA_COL].fillna('').str.contains(q, case=False, na=False) |
                df_search['Indirizzo'].fillna('').str.contains(q, case=False, na=False) |
                df_search['Comune'].fillna('').str.contains(q, case=False, na=False)
            )
            df_search = df_search[mask]
        if region_filter != "(tutte)":
            df_search = df_search[df_search['Regione'] == region_filter]

        if df_search.empty:
            st.info("Nessun Polo Cliente trovato con i criteri inseriti.")
            st.stop()

        # Etichetta completa: Ragione — Indirizzo — Città — Regione
        df_search = df_search.copy()
        df_search['__label'] = (
            df_search[PA_COL].fillna('') + ' — ' +
            df_search['Indirizzo'].fillna('') + ' — ' +
            df_search['Comune'].fillna('') + ' — ' +
            df_search['Regione'].fillna('')
        )
        idx_options = list(range(len(df_search)))
        sel_idx = st.selectbox(
            "Seleziona centro",
            idx_options,
            format_func=lambda i: df_search['__label'].iloc[i],
            key="tab2_sel_idx"
        )
        sel_row = df_search.iloc[sel_idx]
        start_lat, start_lon = float(sel_row['Latitudine']), float(sel_row['Longitudine'])
        start = (start_lat, start_lon)

        # controlli di filtro
        col_a, col_b, col_c = st.columns([2,2,3])
        with col_a:
            time_enable = st.checkbox("Filtra per Tempo (min)", value=True, key="tab2_time_enable")
            time_min = st.slider("Tempo max (min)", 10, 240, 60, 5, disabled=not time_enable, key="tab2_time_min")
        with col_b:
            dist_enable = st.checkbox("Filtra per Distanza (km)", value=True, key="tab2_dist_enable")
            dist_km = st.slider("Distanza max (km)", 5, 300, 50, 5, disabled=not dist_enable, key="tab2_dist_km")
        with col_c:
            region_scope = st.selectbox("Ambito PV", ["Tutta Italia"] + regions, index=0, key="tab2_scope")

        # Ambito PV
        if region_scope != "Tutta Italia":
            df_pv_scope = df_pv_geo[df_pv_geo['Regione'] == region_scope].copy()
        else:
            df_pv_scope = df_pv_geo.copy()

        if df_pv_scope.empty:
            st.warning("Nessun Punto Vendita nell'ambito selezionato.")
            st.stop()

        # --- Distanze Haversine vettoriali (sempre) ---
        lat_pv = pd.to_numeric(df_pv_scope['Latitudine'], errors='coerce').to_numpy(dtype=float)
        lon_pv = pd.to_numeric(df_pv_scope['Longitudine'], errors='coerce').to_numpy(dtype=float)

        R = 6371.0
        lat1 = np.radians(start_lat)
        lon1 = np.radians(start_lon)
        lat2 = np.radians(lat_pv)
        lon2 = np.radians(lon_pv)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        distances_km = 2*R*np.arcsin(np.sqrt(a))   # shape (N,)

        # --- Tempi con ORS (se richiesto e disponibile) ---
        durations_min = None
        if time_enable:
            if ors_client is None:
                st.warning("Filtro Tempo attivo ma chiave ORS assente: verrà ignorato.")
            else:
                try:
                    dests = list(zip(lat_pv, lon_pv))
                    d_m, t_s = _distance_time_matrix(ors_client, start, dests)
                    if t_s is not None:
                        durations_min = (t_s / 60.0).astype(float)  # shape (N,)
                except Exception as e:
                    st.warning(f"Calcolo tempi (ORS) non disponibile: {e}")

        # --- Maschera filtri robusta ---
        N = len(df_pv_scope)
        mask = np.ones(N, dtype=bool)

        if dist_enable:
            mask &= np.isfinite(distances_km) & (distances_km <= float(dist_km))

        if time_enable and durations_min is not None:
            mask &= np.isfinite(durations_min) & (durations_min <= float(time_min))

        # Applica maschera
        result = df_pv_scope[mask].copy()

        # Aggiungi colonne metriche (coerenti col mask)
        result['Distanza_Km'] = distances_km[mask]
        if durations_min is not None:
            result['Tempo_min'] = durations_min[mask]

        # Ordina per Tempo poi Distanza (se presenti)
        sort_cols = [c for c in ['Tempo_min','Distanza_Km'] if c in result.columns]
        if sort_cols:
            result.sort_values(by=sort_cols, inplace=True, ignore_index=True)

        st.subheader("Risultati")
        if result.empty:
            st.warning("Nessun Punto Vendita soddisfa i criteri selezionati.")
        else:
            cols = [c for c in ['Regione','Provincia','Comune','Indirizzo','Tipo Rete','Distanza_Km','Tempo_min'] if c in result.columns]
            # arrotondamenti estetici
            if 'Distanza_Km' in result.columns:
                result['Distanza_Km'] = result['Distanza_Km'].round(2)
            if 'Tempo_min' in result.columns:
                result['Tempo_min'] = result['Tempo_min'].round(1)
            st.dataframe(result[cols], use_container_width=True)

        # Mappa
        st.markdown("---")
        st.subheader("Mappa")
        filtered_ids = result['ID'].astype(str).tolist() if 'ID' in result.columns else []
        show_map(
            df_pa, df_pv, df_conc,
            region_scope if region_scope!="Tutta Italia" else None,
            route_geojson=None,
            filtered_pv_ids=filtered_ids,
            selected_pa=sel_row,
            show_pv=True
        )
