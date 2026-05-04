import os
import numpy as np
import tensorflow as tf
from keras import models
import joblib

def load_assets():
    """Carica il modello e gli scaler dalla cartella corrente."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, "modello_carico.keras")
    scaler_x_path = os.path.join(base_dir, "scaler_X.joblib")
    scaler_y_path = os.path.join(base_dir, "scaler_y.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_x_path, scaler_y_path]):
        raise FileNotFoundError("Errore: Uno o più file del modello (.keras o .joblib) mancano. Esegui prima l'addestramento.")
        
    model = models.load_model(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    return model, scaler_X, scaler_y

def run_prediction(sensor_values, timestamp):
    """
    Esegue la predizione completa partendo dai dati grezzi.
    
    sensor_values: lista di 25 float (w1...w25)
    timestamp: int (0-23)
    """
    # 1. Caricamento Asset
    model, scaler_X, scaler_y = load_assets()
    
    # 2. Preprocessing Temporale (Seni e Coseni)
    periodo = 24
    sin_time = np.sin(2 * np.pi * timestamp / periodo)
    cos_time = np.cos(2 * np.pi * timestamp / periodo)
    
    # 3. Creazione del vettore di input (25 sensori + 2 componenti tempo = 27 feature)
    features = np.array(list(sensor_values) + [sin_time, cos_time]).reshape(1, -1)
    
    # 4. Normalizzazione degli Input
    features_norm = scaler_X.transform(features)
    
    # 5. Predizione (Output normalizzato)
    prediction_norm = model.predict(features_norm, verbose=0)
    
    # 6. De-normalizzazione (Output reale)
    load_reale = scaler_y.inverse_transform(prediction_norm).flatten()[0]
    
    return load_reale

if __name__ == "__main__":
    print("--- ESECUZIONE MODELLO (PREDIZIONE) ---")
    
    # ESEMPIO: Sostituisci questi valori con quelli reali che vuoi testare
    valori_sensori_esempio = [60.0] * 25 # 25 sensori tutti a 60
    ora_esempio = 14
    
    try:
        risultato = run_prediction(valori_sensori_esempio, ora_esempio)
        print(f"\nDati input: Ora {ora_esempio}, Sensori media 60")
        print(f"RISULTATO -> LOAD PREVISTO: {risultato:.2f}")
    except Exception as e:
        print(f"ERRORE: {e}")
