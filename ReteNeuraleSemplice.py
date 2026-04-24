import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import layers, models

# IMPOSTAZIONE RANDOM STATE PER RIPRODUCIBILITÀ COMPLETA
RANDOM_STATE = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forza CPU per evitare non-determinismo GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.config.experimental.enable_op_determinism()

# Disabilita le ottimizzazioni parallele di TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 1. CARICAMENTO DATI
df = pd.read_csv('Dataset/Dataset_random.csv')
df.columns = df.columns.str.strip()

# 2. SELEZIONE CORRETTA DEI SENSORI (Le ultime 25 colonne)
# Se il file ha TIMESTAMP, LOAD e poi 25 sensori (w1...w25)
sensori_cols = df.columns[2:] 
print(f"Sensori selezionati: {sensori_cols.tolist()}") # Verifica che parta da w1

# 3. FEATURE ENGINEERING
# Mediana dei sensori
#df['mediana_sensori'] = df[sensori_cols].median(axis=1)


periodo = 24 
df['sin_time'] = np.sin(2 * np.pi * df['TIMESTAMP'] / periodo)
df['cos_time'] = np.cos(2 * np.pi * df['TIMESTAMP'] / periodo)

# 4. DEFINIZIONE X e y
colonne_totali=sensori_cols.to_list()+['sin_time', 'cos_time']
X = df[colonne_totali].values
y = df['LOAD'].values

# 5. SPLIT E NORMALIZZAZIONE
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6. MODELLO
model = models.Sequential([
    layers.Input(shape=(27,)), 
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# 7. ADDESTRAMENTO
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=32)

# 8. VALUTAZIONE
y_pred = model.predict(X_test)
print(f"\nR^2 Finale: {r2_score(y_test, y_pred):.4f}")

"""Con dataset NON randomicizzato"""
#288 mse 64,32 -->0.8877
#24 mse 64,32 -->0.8868
#288 huber 64,32 -->0.8823
#24 huber 64,32 -->0.8877
#24 log_cosh 64,32 -->0.8878

"""Con dataset randomicizzato"""
#24 huber 64,32 -->0.9412