import sqlite3
import json

# --- Configurazione ---
DB_NAME = 'dati_georoute.db'
TABLE_NAME = 'Anagrafica_benzinai'
JSON_FILE = 'DatasetBenzinai.json' # Assicurati che il nome del file sia corretto!

# --- 1. Definizione dello Schema SQL ---
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    URL TEXT,
    Country TEXT,
    State TEXT,
    City TEXT,
    Station TEXT,
    "EV Charging Points" TEXT, -- <<-- NOTA BENE: I doppi apici qui sono CRUCIALI
    Lat REAL,
    Lon REAL,
    Services TEXT
);
"""

# Le chiavi del JSON nell'ordine originale
JSON_KEYS = [
    "URL", "Country", "State", "City", "Station",
    "EV Charging Points", "Lat", "Lon", "Services"
]

# --- 2. Funzione per l'Importazione Aggiornata ---
def import_json_to_sqlite():
    """
    Legge il file JSON e inserisce i dati nella tabella SQLite.
    Correzione: Gestione dei nomi di colonna con spazi.
    """
    conn = None # Inizializzazione per la clausola finally
    try:
        # Connessione al database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        print(f"Connesso al database: {DB_NAME}")

        # Creazione della tabella (se non esiste)
        cursor.execute(CREATE_TABLE_SQL)
        print(f"Verificata/Creata la tabella: {TABLE_NAME}")

        # Apertura e lettura del file JSON
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Preparazione della query di inserimento

        # *** CORREZIONE QUI ***: Formatta le chiavi del JSON per l'SQL,
        # racchiudendo ogni chiave tra doppi apici per gestire gli spazi.
        sql_columns = [f'"{key}"' for key in JSON_KEYS]
        
        placeholders = ', '.join(['?'] * len(JSON_KEYS))
        insert_sql = f"INSERT INTO {TABLE_NAME} ({', '.join(sql_columns)}) VALUES ({placeholders})"
        
        # Stampa la query per debug (opzionale)
        # print(f"Query di Inserimento: {insert_sql}") 

        # Preparazione dei dati per l'inserimento
        rows_to_insert = []
        # Assumiamo che il file JSON contenga una lista di oggetti
        if isinstance(data, list):
            for item in data:
                # Estrae i valori seguendo l'ordine definito in JSON_KEYS
                values = [item.get(key) for key in JSON_KEYS]
                rows_to_insert.append(tuple(values))
        else:
             # Se il JSON è un singolo oggetto (caso meno comune)
            values = [data.get(key) for key in JSON_KEYS]
            rows_to_insert.append(tuple(values))


        # Esecuzione dell'inserimento multiplo
        if rows_to_insert:
            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            print(f"✅ Importazione completata! Inserite {len(rows_to_insert)} righe.")
        else:
            print("⚠️ Nessun dato trovato nel file JSON da importare.")

    except FileNotFoundError:
        print(f"❌ Errore: File non trovato: {JSON_FILE}")
    except json.JSONDecodeError:
        print(f"❌ Errore: Il file {JSON_FILE} non è un JSON valido.")
    except sqlite3.Error as e:
        print(f"❌ Errore di SQLite: {e}")
    finally:
        # Chiude la connessione
        if conn:
            conn.close()
            print("Connessione al database chiusa.")

# --- 3. Esecuzione ---
if __name__ == "__main__":
    import_json_to_sqlite()