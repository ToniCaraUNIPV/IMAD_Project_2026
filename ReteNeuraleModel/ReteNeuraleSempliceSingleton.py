import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models

# Aggiungiamo la cartella corrente al path per l'importazione del modulo grafici
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ReteNeuraleCreazioneGrafici as graphs

# IMPOSTAZIONE RANDOM STATE PER RIPRODUCIBILITÀ COMPLETA
RANDOM_STATE = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
tf.keras.utils.set_random_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# 1. CARICAMENTO DATI
# Percorso relativo assumendo di eseguire dalla root del progetto
data_path = 'Dataset/Dataset_random.csv'
if not os.path.exists(data_path):
    # Se eseguito dall'interno della cartella ReteNeuraleModel
    data_path = '../Dataset/Dataset_random.csv'

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
y = df['LOAD'].values.reshape(-1, 1) # Reshape per lo scaler

# 5. SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

# 6. NORMALIZZAZIONE (Sia Input che Output)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train_norm = scaler_y.fit_transform(y_train)
y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

# Salvo media e std per la de-normalizzazione manuale se serve, 
# ma userò direttamente lo scaler_y.inverse_transform
y_mean = scaler_y.mean_[0]
y_std = scaler_y.scale_[0]

# 7. MODELLO (Struttura invariata)
initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(len(colonne_totali),)), 
    layers.Dense(64, activation='relu', kernel_initializer=initializer),
    layers.Dense(32, activation='relu', kernel_initializer=initializer),
    layers.Dense(1, kernel_initializer=initializer)
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# 8. ADDESTRAMENTO
print("\nInizio addestramento (con target normalizzato)...")
history = model.fit(
    X_train, y_train_norm, 
    validation_data=(X_val, y_val_norm), 
    epochs=60, 
    batch_size=32, 
    verbose=1
)

# 9. VALUTAZIONE E DE-NORMALIZZAZIONE
print("\nGenerazione previsioni e de-normalizzazione...")
yp_val_norm = model.predict(X_val)
yp_test_norm = model.predict(X_test)

# Torniamo al dominio originale
yp_val = scaler_y.inverse_transform(yp_val_norm).flatten()
yp_test = scaler_y.inverse_transform(yp_test_norm).flatten()
y_val_orig = y_val.flatten()
y_test_orig = y_test.flatten()

# 10. CREAZIONE GRAFICI (Chiamata al modulo esterno)
metrics_val = graphs.evaluate_metrics(y_val_orig, yp_val, "Validation")
metrics_test = graphs.evaluate_metrics(y_test_orig, yp_test, "Test")

# Determiniamo il percorso di salvataggio nella stessa cartella dello script
save_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(save_dir, "rete_singleton_results.png")

graphs.plot_results(
    history, 
    y_val_orig, yp_val, 
    y_test_orig, yp_test, 
    metrics_val, metrics_test,
    save_path=save_path
)

print(f"\nModello addestrato e grafici salvati in: {save_path}")
