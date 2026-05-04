import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models
import joblib

# Aggiungiamo la cartella superiore (ReteNeuraleModel) al path per l'importazione
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ReteNeuraleCreazioneGrafici as graphs

# IMPOSTAZIONE RANDOM STATE PER RIPRODUCIBILITÀ COMPLETA
RANDOM_STATE = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
tf.keras.utils.set_random_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def train_and_save_model():
    """
    Funzione che carica i dati, addestra la rete, salva i grafici 
    e genera i file .keras e .joblib necessari per la predizione.
    """
    # 1. CARICAMENTO DATI (Cerca il dataset in base alla posizione dello script)
    base_script_path = os.path.dirname(os.path.abspath(__file__))
    # Percorsi possibili: dalla root o risalendo di due cartelle
    data_options = [
        os.path.join(base_script_path, "../../Dataset/Dataset_random.csv"),
        'Dataset/Dataset_random.csv',
        '../Dataset/Dataset_random.csv'
    ]
    
    data_path = None
    for path in data_options:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        raise FileNotFoundError("Impossibile trovare Dataset/Dataset_random.csv. Verifica la posizione!")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # 2. SELEZIONE SENSORI
    sensori_cols = df.columns[2:] 

    # 3. FEATURE ENGINEERING
    periodo = 24 
    df['sin_time'] = np.sin(2 * np.pi * df['TIMESTAMP'] / periodo)
    df['cos_time'] = np.cos(2 * np.pi * df['TIMESTAMP'] / periodo)

    # 4. DEFINIZIONE X e y
    colonne_totali = sensori_cols.to_list() + ['sin_time', 'cos_time']
    X = df[colonne_totali].values
    y = df['LOAD'].values.reshape(-1, 1)

    # 5. SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

    # 6. NORMALIZZAZIONE
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)

    # 7. MODELLO
    initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_STATE)
    model = models.Sequential([
        layers.Input(shape=(len(colonne_totali),)), 
        layers.Dense(64, activation='relu', kernel_initializer=initializer),
        layers.Dense(32, activation='relu', kernel_initializer=initializer),
        layers.Dense(1, kernel_initializer=initializer)
    ])

    model.compile(optimizer='adam', loss='huber', metrics=['mae'])

    # 8. ADDESTRAMENTO
    print("\nInizio addestramento...")
    history = model.fit(
        X_train, y_train_norm, 
        validation_data=(X_val, y_val_norm), 
        epochs=60, 
        batch_size=32, 
        verbose=1
    )

    # 9. SALVATAGGIO ASSET
    save_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(save_dir, "modello_carico.keras"))
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.joblib"))
    joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.joblib"))
    print(f"\n[OK] Asset salvati in {save_dir}")

    # 10. GRAFICI
    yp_val = scaler_y.inverse_transform(model.predict(X_val)).flatten()
    yp_test = scaler_y.inverse_transform(model.predict(X_test)).flatten()
    metrics_val = graphs.evaluate_metrics(y_val.flatten(), yp_val, "Validation")
    metrics_test = graphs.evaluate_metrics(y_test.flatten(), yp_test, "Test")
    graphs.plot_results(history, y_val.flatten(), yp_val, y_test.flatten(), yp_test, 
                        metrics_val, metrics_test, save_path=os.path.join(save_dir, "results_production.png"))

if __name__ == "__main__":
    train_and_save_model()
