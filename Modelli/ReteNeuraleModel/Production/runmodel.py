import os
import sys
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Esegue la predizione del LOAD dai dati dei sensori.")
    parser.add_argument("timestamp", type=int, help="L'ora del giorno (0-23)")
    parser.add_argument("sensors", type=float, nargs=25, help="I 25 valori dei sensori (w1...w25) separati da spazio")
    
    # Se non vengono passati argomenti, mostra l'aiuto o esegui un esempio
    if len(sys.argv) == 1:
        print("--- ESECUZIONE MODELLO (ESEMPIO) ---")
        print("Uso: python runmodel.py <ora> <w1> <w2> ... <w25>")
        valori_sensori_esempio = [60.0] * 25
        ora_esempio = 14
        try:
            risultato = run_prediction(valori_sensori_esempio, ora_esempio)
            print(f"\nEsempio -> Ora {ora_esempio}, Sensori media 60")
            print(f"LOAD PREVISTO: {risultato:.2f}")
        except Exception as e:
            print(f"ERRORE: {e}")
    else:
        args = parser.parse_args()
        try:
            risultato = run_prediction(args.sensors, args.timestamp)
            print(f"\nPrevisione per Ora {args.timestamp}:")
            print(f"LOAD -> {risultato:.2f}")
        except Exception as e:
            print(f"ERRORE durante la predizione: {e}")
