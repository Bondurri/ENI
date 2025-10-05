
# foresee.py
# -*- coding: utf-8 -*-
"""
Foresee – Sperimentazione: trovare le migliori location per massimizzare la copertura dei Punti Vendita
- Parte 1: individua i migliori PoliAmbulatori per vicinanza dei PuntiVendita
- Parte 2: individua location teoriche (Lat/Lon) che coprano almeno N=20/30/40 PV per zona,
           poi confronta con il PoliAmbulatori più vicino.
Note: usa solo Haversine (km) per le distanze; nessuna dipendenza da servizi esterni.
Formattazione italiana: i valori mostrati nelle tabelle usano la virgola decimale.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import FastMarkerCluster


# ------------------------------------------------------------------------------
# Config / Costanti
# ------------------------------------------------------------------------------
PA_COL = "Ragione sociale Fornitore"   # nome canonico per il Polo/Cliente
R_EARTH_KM = 6371.0


# ------------------------------------------------------------------------------
# Utilità dati / normalizzazioni
# ------------------------------------------------------------------------------
def ensure_pa_name_col(df: pd.DataFrame) -> pd.DataFrame:
    """Se non esiste PA_COL, rinomina automaticamente un sinonimo a PA_COL."""
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
    return df


def normalize_region(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and 'Regione' in df.columns:
        df['Regione'] = (
            df['Regione'].astype(str).str.strip().str.upper()
              .str.replace('-', ' ', regex=False)
              .str.replace(r'\s+', ' ', regex=True)
        )
    return df


def get_regions(df_pa: pd.DataFrame, df_pv: pd.DataFrame) -> List[str]:
    reg = []
    if 'Regione' in df_pa.columns:
        reg += df_pa['Regione'].dropna().astype(str).tolist()
    if 'Regione' in df_pv.columns:
        reg += df_pv['Regione'].dropna().astype(str).tolist()
    return sorted(list(set(reg)))


# ------------------------------------------------------------------------------
# Formattazione IT (virgola decimale)
# ------------------------------------------------------------------------------
def fmt_it(value, dec: int = 2) -> str:
    """Formatta un numero in stile italiano: 12.345,67"""
    try:
        x = float(value)
    except Exception:
        return "—"
    # Usa formattazione US, poi sostituisci separatori
    s = f"{x:,.{dec}f}"  # 12,345.67
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def fmt_series_it(s: pd.Series, dec: int = 2) -> pd.Series:
    return s.apply(lambda v: fmt_it(v, dec))


# ------------------------------------------------------------------------------
# Distanze Haversine (km) – vettoriali
# ------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Accetta scalari o array numpy coerenti."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------------------
# Parte 1: Migliori PoliAmbulatori per vicinanza
# ------------------------------------------------------------------------------
def rank_poli_by_nearest(df_pa: pd.DataFrame, df_pv: pd.DataFrame) -> pd.DataFrame:
    """
    Assegna ogni PV al PA più vicino (in km) e calcola: N clienti, somma km, media km, mediana km.
    Ritorna un DataFrame ordinato per N clienti desc, poi media km asc.
    """
    if df_pa.empty or df_pv.empty:
        return pd.DataFrame()

    lat_pa = df_pa['Latitudine'].to_numpy(dtype=float)
    lon_pa = df_pa['Longitudine'].to_numpy(dtype=float)
    lat_pv = df_pv['Latitudine'].to_numpy(dtype=float)
    lon_pv = df_pv['Longitudine'].to_numpy(dtype=float)

    # Matrice distanze PV x PA (potrebbe essere grande, usiamo batching se serve)
    B = 2000  # batch PV per contenere memoria
    nearest_idx = []
    nearest_dist = []
    for i in range(0, len(lat_pv), B):
        latb = lat_pv[i:i+B][:, None]  # (b,1)
        lonb = lon_pv[i:i+B][:, None]
        # distanza verso tutti PA (b, M)
        d = haversine_km(latb, lonb, lat_pa[None, :], lon_pa[None, :])
        idx = np.argmin(d, axis=1)
        nearest_idx.append(idx)
        nearest_dist.append(d[np.arange(d.shape[0]), idx])
    nearest_idx = np.concatenate(nearest_idx) if nearest_idx else np.array([], dtype=int)
    nearest_dist = np.concatenate(nearest_dist) if nearest_dist else np.array([], dtype=float)

    # Aggrega per PA
    pa_names = df_pa[PA_COL].astype(str).tolist()
    pa_regions = df_pa['Regione'].astype(str).tolist() if 'Regione' in df_pa.columns else [""]*len(df_pa)

    agg = {}
    for pv_i, pa_i in enumerate(nearest_idx):
        agg.setdefault(pa_i, []).append(nearest_dist[pv_i])

    rows = []
    for pa_i, dists in agg.items():
        arr = np.array(dists, dtype=float)
        rows.append({
            "PA_index": pa_i,
            "Polo": pa_names[pa_i] if pa_i < len(pa_names) else f"PA_{pa_i}",
            "Regione": pa_regions[pa_i] if pa_i < len(pa_regions) else "",
            "N_clienti": len(arr),
            "Somma_km": float(arr.sum()),
            "Media_km": float(arr.mean()),
            "Mediana_km": float(np.median(arr)),
        })

    df_rank = pd.DataFrame(rows)
    if df_rank.empty:
        return df_rank

    # Aggiungi lat/lon del PA
    df_rank['Latitudine'] = [lat_pa[i] for i in df_rank['PA_index']]
    df_rank['Longitudine'] = [lon_pa[i] for i in df_rank['PA_index']]

    df_rank.sort_values(by=['N_clienti', 'Media_km'], ascending=[False, True], inplace=True, ignore_index=True)
    return df_rank


# ------------------------------------------------------------------------------
# Parte 2: Cluster "almeno N per zona" + centro ottimo
# ------------------------------------------------------------------------------
def grow_clusters_min_size(lat: np.ndarray, lon: np.ndarray, min_size: int) -> List[np.ndarray]:
    """
    Clustering greedy: crea cluster che contengono ALMENO `min_size` punti.
    Strategie:
    - Finché restano punti non assegnati, prendi un seed (quello più lontano dalla media
      dei restanti) e aggiungi i (min_size-1) più vicini a quel seed.
    - L'ultimo cluster, se < min_size, viene fuso col cluster più vicino.
    Ritorna lista di array di indici (riferiti all'array originale).
    """
    n = len(lat)
    remaining = np.arange(n)
    clusters: List[np.ndarray] = []

    while len(remaining) > 0:
        if len(remaining) <= min_size:
            clusters.append(remaining.copy())
            break

        # seed: punto più distante dal baricentro dei rimanenti (euristica)
        lat_r = lat[remaining]; lon_r = lon[remaining]
        centroid_lat = lat_r.mean(); centroid_lon = lon_r.mean()
        d2c = haversine_km(lat_r, lon_r, centroid_lat, centroid_lon)  # (m,)
        seed_rel = int(np.argmax(d2c))
        seed_idx = remaining[seed_rel]

        # vicini al seed
        dseed = haversine_km(lat_r, lon_r, lat[seed_idx], lon[seed_idx])  # distanze dal seed ai rimanenti
        order = np.argsort(dseed)  # dal più vicino
        take_rel = order[:min_size]  # almeno min_size
        cluster = remaining[take_rel]

        clusters.append(cluster)
        # togli assegnati
        mask = np.ones(len(remaining), dtype=bool)
        mask[take_rel] = False
        remaining = remaining[mask]

    # Se l'ultimo cluster è piccolo, fondi col più vicino
    if len(clusters) >= 2 and len(clusters[-1]) < min_size:
        small = clusters.pop(-1)
        # trova cluster più vicino all'ultimo
        cent_small_lat = lat[small].mean(); cent_small_lon = lon[small].mean()
        best_j = None; best_d = float('inf')
        for j, cl in enumerate(clusters):
            cent_lat = lat[cl].mean(); cent_lon = lon[cl].mean()
            dj = float(haversine_km(cent_small_lat, cent_small_lon, cent_lat, cent_lon))
            if dj < best_d:
                best_d = dj; best_j = j
        if best_j is not None:
            clusters[best_j] = np.concatenate([clusters[best_j], small])

    return clusters


def spherical_centroid(lat: np.ndarray, lon: np.ndarray) -> Tuple[float, float]:
    """Centroid su sfera (media vettoriale)."""
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    x_m, y_m, z_m = x.mean(), y.mean(), z.mean()
    lon_c = math.atan2(y_m, x_m)
    hyp = math.sqrt(x_m*x_m + y_m*y_m)
    lat_c = math.atan2(z_m, hyp)
    return (math.degrees(lat_c), math.degrees(lon_c))


def geometric_median_haversine(lat: np.ndarray, lon: np.ndarray, tol: float = 1e-6, max_iter: int = 200) -> Tuple[float, float]:
    """
    Weiszfeld su piano tangente (approssimazione robusta per coordinate geografiche).
    Proietta localmente in metri usando equirettangolare centrata sul baricentro.
    """
    lat0, lon0 = spherical_centroid(lat, lon)
    # proiezione equirettangolare locale
    phi0 = math.radians(lat0); lam0 = math.radians(lon0)
    R = 6371000.0  # m
    x = R * (np.radians(lon) - lam0) * math.cos(phi0)
    y = R * (np.radians(lat) - phi0)

    # Weiszfeld
    xk = x.mean(); yk = y.mean()
    for _ in range(max_iter):
        dx = x - xk; dy = y - yk
        dist = np.sqrt(dx*dx + dy*dy) + 1e-12
        w = 1.0 / dist
        x_new = np.sum(w * x) / np.sum(w)
        y_new = np.sum(w * y) / np.sum(w)
        if math.hypot(x_new - xk, y_new - yk) < tol:
            xk, yk = x_new, y_new
            break
        xk, yk = x_new, y_new

    # back-projection
    lat_m = math.degrees(yk / R + phi0)
    lon_m = math.degrees(xk / (R * math.cos(phi0)) + lam0)
    return (lat_m, lon_m)


def build_cluster_candidates(df_pv: pd.DataFrame, min_size: int, center_method: str = "geomedian") -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Crea cluster di Punti di Interessi con dimensione minima e calcola per ciascuno:
      - centro candidato (Lat/Lon) come 'geometric median' o 'spherical centroid'
      - statistiche (somma/medio delle distanze dal centro)
    Ritorna: (df_clusters, clusters_indices)
    """
    lat = pd.to_numeric(df_pv['Latitudine'], errors='coerce').to_numpy(dtype=float)
    lon = pd.to_numeric(df_pv['Longitudine'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    idx_all = np.where(valid)[0]
    lat = lat[valid]; lon = lon[valid]

    clusters = grow_clusters_min_size(lat, lon, min_size)
    rows = []
    centroids = []
    for k, cl_idx_rel in enumerate(clusters, start=1):
        cl_abs = idx_all[cl_idx_rel]
        lat_cl = lat[cl_idx_rel]; lon_cl = lon[cl_idx_rel]

        if center_method == "centroid":
            latc, lonc = spherical_centroid(lat_cl, lon_cl)
        else:
            latc, lonc = geometric_median_haversine(lat_cl, lon_cl)

        # metriche distanza dal centro
        d = haversine_km(lat_cl, lon_cl, latc, lonc)
        rows.append({
            "ClusterID": k,
            "N_PV": int(len(cl_abs)),
            "Lat_centro": float(latc),
            "Lon_centro": float(lonc),
            "Somma_km": float(d.sum()),
            "Media_km": float(d.mean()),
            "Mediana_km": float(np.median(d)),
        })
        centroids.append((latc, lonc))

    df_clusters = pd.DataFrame(rows)
    return df_clusters, clusters


def nearest_pa_to_points(df_pa: pd.DataFrame, points_lat: np.ndarray, points_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per ogni punto (lat,lon), trova il PoliAmbulatorio più vicino. Ritorna (idx_pa, dist_km).
    """
    lat_pa = df_pa['Latitudine'].to_numpy(dtype=float)
    lon_pa = df_pa['Longitudine'].to_numpy(dtype=float)
    M = len(lat_pa)
    idx_ret = np.zeros(len(points_lat), dtype=int)
    dist_ret = np.zeros(len(points_lat), dtype=float)

    # batch per memoria
    B = 2048
    for i in range(0, len(points_lat), B):
        la = points_lat[i:i+B][:, None]
        lo = points_lon[i:i+B][:, None]
        d = haversine_km(la, lo, lat_pa[None, :], lon_pa[None, :])  # (b,M)
        idx = np.argmin(d, axis=1)
        idx_ret[i:i+B] = idx
        dist_ret[i:i+B] = d[np.arange(d.shape[0]), idx]

    return idx_ret, dist_ret


# ------------------------------------------------------------------------------
# Mappe
# ------------------------------------------------------------------------------
def draw_cluster_map(df_pv_cluster, lat_c, lon_c, df_pa_f, pa_i=None, pa_d=None, k_nearest_pamb: int = 2):
    """
    Disegna la mappa del cluster:
      - PI del cluster in BLU
      - Centro teorico in ROSSO
      - PAmb esistenti in ARANCIONE (solo i k_nearest_pamb più vicini al centro)
    Ritorna: lista degli indici (rispetto a df_pa_f) dei PAmb effettivamente mostrati.
    """
    import numpy as np
    import folium
    from streamlit_folium import st_folium

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2.0)**2)
        c = 2*np.arcsin(np.sqrt(a))
        return R * c

    m = folium.Map(location=[lat_c, lon_c], zoom_start=8)

    # --- PI del cluster (blu) ---
    for _, r in df_pv_cluster.iterrows():
        lat, lon = r.get('Latitudine'), r.get('Longitudine')
        if lat is not None and lon is not None and not (np.isnan(lat) or np.isnan(lon)):
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.6,
                popup=str(r.get("Comune", "PI"))
            ).add_to(m)

    # --- Centro teorico (rosso) ---
    folium.Marker(
        location=[lat_c, lon_c],
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
        popup="Centro teorico del cluster"
    ).add_to(m)

    # --- PAmb esistenti: prendiamo SOLO i k più vicini al centro ---
    plotted_idx = []
    if df_pa_f is not None and not df_pa_f.empty and k_nearest_pamb > 0:
        # calcola distanze dei PAmb al centro
        lat_arr = df_pa_f['Latitudine'].astype(float).to_numpy()
        lon_arr = df_pa_f['Longitudine'].astype(float).to_numpy()
        mask_valid = ~(np.isnan(lat_arr) | np.isnan(lon_arr))
        dists = np.full(len(df_pa_f), np.inf, dtype=float)
        dists[mask_valid] = haversine_km(lat_arr[mask_valid], lon_arr[mask_valid], lat_c, lon_c)

        # ordina per distanza e prendi i primi k
        order = np.argsort(dists)
        take = [i for i in order[:int(k_nearest_pamb)] if np.isfinite(dists[i])]

        for i in take:
            r = df_pa_f.iloc[i]
            folium.Marker(
                location=[float(r['Latitudine']), float(r['Longitudine'])],
                icon=folium.Icon(color="orange", icon="plus-square", prefix="fa"),
                popup=str(r.get(PA_COL, "PoliAmbulatorio"))
            ).add_to(m)
        plotted_idx = take

    st_folium(m, width=1000, height=600)
    return plotted_idx




# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
import streamlit as st
import numpy as np

def render_foresee_ui():
    # --- Stile coerente con app.py ---
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

    st.title("🔮 Foresee – Ottimizzazione Location per Copertura Punti di Interesse")

    # --- Recupero dati dalla sessione ---
    df_pa = st.session_state.get("df_pa")   # PoliAmbulatori
    df_pv = st.session_state.get("df_pv")   # Punti di Interesse

    if df_pa is None or df_pv is None:
        st.error("⚠️ Nessun dato disponibile. Carica prima PoliAmbulatori e Punti di Interesse nella sezione Geolocalizzazione.")
        return

    # Normalizza nomi colonne e regioni
    df_pa = ensure_pa_name_col(df_pa)
    df_pa = normalize_region(df_pa)
    df_pv = normalize_region(df_pv)

    # --------------------------------------------------------------------------
    # Filtri base
    # --------------------------------------------------------------------------
    regions = get_regions(df_pa, df_pv)
    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        region = st.selectbox("Filtro Regione (opzionale)", ["(tutte)"] + regions, index=0)
    with colB:
        min_size = st.number_input("Dimensione min.Cluster (numero min.Punti di Interesse)", 
                                   min_value=5, max_value=200, value=20, step=1)
    with colC:
        center_method = st.selectbox(
            "Centro cluster",
            ["geomedian", "centroid", "tempo_percorrenza", "traffico_pesato", "ibrido"],
            index=0,
            help="Geomedian = mediana geometrica; Centroid = baricentro; "
                 "Tempo_percorrenza = centro che minimizza i tempi reali di spostamento; "
                 "Traffico_pesato = centro che minimizza tempi pesati per traffico medio; "
                 "Ibrido = combinazione di tempo di percorrenza e traffico."
        )

    # Applica filtro regione
    df_pa_f = df_pa.copy()
    df_pv_f = df_pv.copy()
    if region != "(tutte)" and 'Regione' in df_pv.columns:
        df_pv_f = df_pv_f[df_pv_f['Regione'] == region].copy()
    if region != "(tutte)" and 'Regione' in df_pa.columns:
        df_pa_f = df_pa_f[df_pa_f['Regione'] == region].copy()

    # 🔧 Importante: rinomina eventuale colonna nome PA
    df_pa_f = ensure_pa_name_col(df_pa_f)

    # Check colonne minime
    needed_cols = ['Latitudine', 'Longitudine']
    ok = True
    for dname, d in [('Poli Ambulatori', df_pa_f), ('Punti di Interesse', df_pv_f)]:
        for c in needed_cols:
            if c not in d.columns:
                st.error(f"{dname}: manca la colonna '{c}'.")
                ok = False
    if PA_COL not in df_pa_f.columns:
        st.warning(f"Poli Ambulatori: non trovo '{PA_COL}'. Rinominare una colonna esistente o assicurarsi che la normalizzazione funzioni.")
    if not ok:
        st.stop()

    # --------------------------------------------------------------------------
    # PARTE 1 — Migliori PoliAmbulatori
    # --------------------------------------------------------------------------
    st.header("1) Migliori PoliAmbulatori per vicinanza Punti di Interesse")
    if df_pa_f.empty or df_pv_f.empty:
        st.info("Carica Poli Ambulatori e Punti di Interesse per proseguire.")
    else:
        df_rank = rank_poli_by_nearest(df_pa_f, df_pv_f)
        if df_rank.empty:
            st.info("Nessuna assegnazione possibile.")
        else:
            show_cols = ['Polo','Regione','N_clienti','Media_km','Mediana_km','Somma_km','Latitudine','Longitudine']
            tbl = df_rank[show_cols].copy()
            tbl['Media_km']   = fmt_series_it(tbl['Media_km'],   2)
            tbl['Mediana_km'] = fmt_series_it(tbl['Mediana_km'], 2)
            tbl['Somma_km']   = fmt_series_it(tbl['Somma_km'],   2)
            tbl['Latitudine']  = fmt_series_it(df_rank['Latitudine'], 5)
            tbl['Longitudine'] = fmt_series_it(df_rank['Longitudine'], 5)

            st.dataframe(tbl, use_container_width=True)
            st.caption("Classifica per numero di Punti di Interesse assegnati (desc) e distanza media (asc).")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # PARTE 2 — Cluster Location Teoriche
    # --------------------------------------------------------------------------
    st.header(f"2) Location teoriche: cluster con almeno {min_size} Punti di Interesse")
    if df_pv_f.empty:
        st.info("Servono i Punti di Interesse per creare i cluster.")
        st.stop()

    # Qui usi comunque build_cluster_candidates standard,
    # ma in futuro potrai implementare logiche diverse per "tempo_percorrenza", "traffico_pesato" e "ibrido".
    df_clusters, clusters_idx = build_cluster_candidates(df_pv_f, int(min_size), center_method=center_method)
    if df_clusters.empty:
        st.info("Nessun cluster creato.")
        st.stop()

    # Trova PoliAmbulatori più vicini ai centri
    latc = df_clusters['Lat_centro'].to_numpy(dtype=float)
    lonc = df_clusters['Lon_centro'].to_numpy(dtype=float)
    if df_pa_f.empty:
        idx_pa = np.array([-1]*len(df_clusters), dtype=int)
        dist_pa = np.array([np.nan]*len(df_clusters), dtype=float)
        pa_names = ["—"]*len(df_clusters)
    else:
        idx_pa, dist_pa = nearest_pa_to_points(df_pa_f, latc, lonc)
        pa_names = [df_pa_f.iloc[i][PA_COL] if (i>=0 and i<len(df_pa_f)) else "—" for i in idx_pa]

    df_clusters['PAmb_vicino'] = pa_names
    df_clusters['PAmb_dist_km'] = dist_pa

    # 🔧 Nomi cluster regionali
    cluster_names = []
    for i, idx in enumerate(clusters_idx, start=1):
        if len(idx) > 0 and "Regione" in df_pv_f.columns:
            region_vals = df_pv_f.iloc[idx]["Regione"].dropna().unique().tolist()
            if len(region_vals) == 1:
                reg = region_vals[0]
            elif len(region_vals) > 1:
                reg = region_vals[0] + "+"
            else:
                reg = "?"
            cluster_names.append(f"{reg}{i}")
        else:
            cluster_names.append(f"Cluster{i}")

    df_clusters["ClusterName"] = cluster_names

    st.subheader("Riepilogo cluster teorici")
    tblc = df_clusters[['ClusterName','N_PV','Lat_centro','Lon_centro','Media_km','Mediana_km','Somma_km','PAmb_vicino','PAmb_dist_km']].copy()
    for col, dec in [('Lat_centro',5), ('Lon_centro',5), ('Media_km',2), ('Mediana_km',2), ('Somma_km',2), ('PAmb_dist_km',1)]:
        tblc[col] = fmt_series_it(tblc[col], dec)
    st.dataframe(tblc, use_container_width=True)

    # Mappa cluster selezionato
    st.subheader("Mappa cluster selezionato")
    cl_options = df_clusters['ClusterName'].tolist()
    sel_cl = st.selectbox("Seleziona cluster", cl_options, index=0)
    row = df_clusters[df_clusters['ClusterName'] == sel_cl].iloc[0]
    sel_idx = df_clusters.index[df_clusters['ClusterName'] == sel_cl][0]
    idx_rel = clusters_idx[sel_idx]
    df_pv_cluster = df_pv_f.iloc[idx_rel].copy()

    pa_i = int(idx_pa[sel_idx]) if len(idx_pa) == len(df_clusters) else None
    pa_d = float(dist_pa[sel_idx]) if len(dist_pa) == len(df_clusters) else None

    # Disegna mappa e RITORNA gli indici dei PAmb realmente mostrati in arancione
    shown_pa_indices = draw_cluster_map(
        df_pv_cluster,
        float(row['Lat_centro']),
        float(row['Lon_centro']),
        df_pa_f,
        pa_i,
        pa_d,
        k_nearest_pamb=2  # <-- se vuoi cambiarne il numero, modificalo qui o metti un controllo UI
    )

    # ----------------------------------------------------------------------
    # Spiegazione generale
    # ----------------------------------------------------------------------
    st.markdown("### 📍 Spiegazione generale")
    st.write(
        "Il punto rosso mostrato sulla mappa rappresenta una **locazione teorica**, "
        "calcolata per massimizzare la copertura dei Punti di Interesse (PI) vicini. "
        "Non corrisponde a un luogo reale, ma indica **dove converrebbe posizionare un nuovo PoliAmbulatorio** "
        "per attrarre il maggior numero di PI, riducendo le distanze medie."
    )
    st.write(
        "I PoliAmbulatori già esistenti sono mostrati in **arancione**, mentre i PI del cluster selezionato sono in **blu**."
    )

    # Wrapper espandibile: Nomi PAmb ESISTENTI mostrati in mappa (arancione)
    with st.expander("📋 Nomi PoliAmbulatori esistenti in mappa (arancione)"):
        if shown_pa_indices:
            names = df_pa_f.iloc[shown_pa_indices][PA_COL].dropna().astype(str).tolist()
            st.markdown("\n".join(f"- {n}" for n in names))
        else:
            st.markdown("_Nessun PoliAmbulatorio mostrato in mappa._")

    # Wrapper espandibile: Nomi PI del cluster (blu)
    with st.expander("📋 Nomi Punti di Interesse del cluster (blu)"):
        if not df_pv_cluster.empty:
            if "Comune" in df_pv_cluster.columns:
                items = df_pv_cluster["Comune"].dropna().astype(str).unique().tolist()
                st.markdown("\n".join(f"- {c}" for c in items))
            else:
                # fallback: se non c'è 'Comune', elenca con indice
                items = df_pv_cluster.index.astype(str).tolist()
                st.markdown("\n".join(f"- {c}" for c in items))
        else:
            st.markdown("_Nessun PI nel cluster selezionato._")

    # Wrapper espandibile: spiegazione algoritmi
    with st.expander("⚙️ Spiegazione algoritmi clustering"):
        st.markdown("**Geomedian (mediana geometrica)**")
        st.write(
            "Il centro del cluster è calcolato come il punto che minimizza la somma "
            "delle distanze da tutti i Punti di Interesse. È più robusto rispetto agli "
            "outlier (punti molto lontani), ma richiede calcoli più complessi."
        )

        st.markdown("**Centroid (baricentro)**")
        st.write(
            "Il centro del cluster è calcolato come la media aritmetica delle coordinate "
            "di tutti i Punti di Interesse. È semplice e veloce da calcolare, "
            "ma può essere influenzato da punti estremi."
        )

    
        st.markdown("**Centro minimo tempo di percorrenza**")
        st.write(
            "Il centro è scelto per minimizzare il **tempo medio di spostamento** "
            "tra tutti i PI e il centro stesso, utilizzando dati di viabilità reale."
        )

        st.markdown("**Centro pesato sul traffico**")
        st.write(
            "Il centro è calcolato tenendo conto del **traffico medio** sulle strade. "
            "Le zone più congestionate hanno peso maggiore e spostano il centro verso aree "
            "più facilmente raggiungibili."
        )
    # ----------------------------------------------------------------------
    # Spiegazione del cluster selezionato
    # ----------------------------------------------------------------------
    st.markdown("### ℹ️ Spiegazione del cluster selezionato")

    st.markdown("**a) Cos’è**")
    st.write(
        "Un *cluster* rappresenta un gruppo di Punti di Interesse (PI) vicini tra loro, "
        "aggregati per individuare una possibile area di copertura comune."
    )

    st.markdown("**b) Come si è ottenuto**")
    st.write(
        f"I {len(df_pv_cluster)} PI di questo cluster appartengono "
        f"alla regione {', '.join(df_pv_cluster['Regione'].dropna().unique().tolist()) if 'Regione' in df_pv_cluster.columns else 'sconosciuta'}. "
        f"Sono stati raggruppati con un algoritmo di clustering spaziale, "
        f"imponendo una soglia minima di **{min_size} PI** scelta dall’utente."
    )

    st.markdown("**c) Algoritmi usati**")
    st.write(
        f"Centro calcolato con metodo **{center_method}**. "
        "• *Geomedian* = mediana geometrica, robusta agli outlier. "
        "• *Centroid* = baricentro (media) delle coordinate geografiche."
    )

    st.markdown("**d) Eventuali dati mancanti**")
    if df_pa_f.empty:
        st.warning("⚠️ Non sono disponibili PoliAmbulatori in questa regione, quindi non è stato possibile calcolare le distanze verso i PAmb.")
    elif any(df_pv_cluster[['Latitudine','Longitudine']].isna().any()):
        st.warning("⚠️ Alcuni PI del cluster non hanno coordinate valide (Latitudine/Longitudine mancanti).")
    else:
        st.success("✅ Tutti i dati necessari erano disponibili per il calcolo.")



# Entrypoint Streamlit
if __name__ == "__main__":
    render_foresee_ui()
