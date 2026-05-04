import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# IMPOSTAZIONE RANDOM STATE PER RIPRODUCIBILITÀ COMPLETA
RANDOM_STATE = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forza CPU per evitare non-determinismo GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.keras.utils.set_random_seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
# tf.config.experimental.enable_op_determinism() # Può causare problemi su alcune versioni di TF se non supportato

# Disabilita le ottimizzazioni parallele di TensorFlow per riproducibilità
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 1. CARICAMENTO DATI
df = pd.read_csv('Dataset/Dataset_random.csv')
df.columns = df.columns.str.strip()

# 2. SELEZIONE CORRETTA DEI SENSORI (Le ultime 25 colonne)
sensori_cols = df.columns[2:] 
print(f"Sensori selezionati: {sensori_cols.tolist()}") 

# 3. FEATURE ENGINEERING
periodo = 24 
df['sin_time'] = np.sin(2 * np.pi * df['TIMESTAMP'] / periodo)
df['cos_time'] = np.cos(2 * np.pi * df['TIMESTAMP'] / periodo)

# 4. DEFINIZIONE X e y
colonne_totali = sensori_cols.to_list() + ['sin_time', 'cos_time']
X = df[colonne_totali].values
y = df['LOAD'].values

# 5. SPLIT E NORMALIZZAZIONE
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6. MODELLO (Invariato rispetto a ReteNeuraleSemplice.py)
initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(len(colonne_totali),)), 
    layers.Dense(64, activation='relu', kernel_initializer=initializer),
    layers.Dense(32, activation='relu', kernel_initializer=initializer),
    layers.Dense(1, kernel_initializer=initializer)
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# 7. ADDESTRAMENTO
print("\nInizio addestramento...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=32, verbose=1)

# ══════════════════════════════════════════════════════════════════════
# FUNZIONI DI VALUTAZIONE E VISUALIZZAZIONE (Ispirate a load_cnn_lstm.py)
# ══════════════════════════════════════════════════════════════════════

def evaluate_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    print(f"\n{'─'*42}")
    print(f"  {label}")
    print(f"{'─'*42}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)

def plot_results(history,
                 y_val,  yp_val,
                 y_test, yp_test,
                 metrics_val, metrics_test,
                 save_path="rete_semplice_results.png"):

    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    # ── [0,0-1] Loss curve ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(history.history["loss"],     label="Train loss", linewidth=1.5)
    ax.plot(history.history["val_loss"], label="Val loss",   linewidth=1.5)
    ax.set_title("Huber Loss — Training / Validation", fontsize=13)
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # ── [1, 0] Predizioni Validation (Subset di campioni) ────────────────
    ax = fig.add_subplot(gs[1, 0])
    n  = min(168, len(y_val)) # Mostriamo 168 campioni (equivalente a 1 settimana se fossero sequenziali)
    ax.plot(y_val[:n],  label="Reale",    linewidth=1.2)
    ax.plot(yp_val[:n], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title(f"Validation — Primi {n} campioni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [1, 1] Predizioni Test (Subset di campioni) ──────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n  = min(168, len(y_test))
    ax.plot(y_test[:n],  label="Reale",    linewidth=1.2)
    ax.plot(yp_test[:n], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title(f"Test — Primi {n} campioni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 0] Scatter Val: Reale vs Predetto ────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(y_val, yp_val, alpha=0.15, s=4, color="steelblue")
    mn, mx = min(y_val.min(), yp_val.min()), max(y_val.max(), yp_val.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_title(f"Validation — Reale vs Predetto  (R²={metrics_val['r2']:.4f})", fontsize=12)
    ax.set_xlabel("LOAD Reale"); ax.set_ylabel("LOAD Predetto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 1] Scatter Test: Reale vs Predetto ───────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(y_test, yp_test, alpha=0.15, s=4, color="darkorange")
    mn, mx = min(y_test.min(), yp_test.min()), max(y_test.max(), yp_test.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_title(f"Test — Reale vs Predetto  (R²={metrics_test['r2']:.4f})", fontsize=12)
    ax.set_xlabel("LOAD Reale"); ax.set_ylabel("LOAD Predetto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [3, :] Tabella metriche Val vs Test ───────────────────────────
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    labels  = ["MAE", "RMSE", "MAPE (%)", "R²"]
    v_vals  = [f"{metrics_val['mae']:.3f}",
               f"{metrics_val['rmse']:.3f}",
               f"{metrics_val['mape']:.2f}",
               f"{metrics_val['r2']:.4f}"]
    t_vals  = [f"{metrics_test['mae']:.3f}",
               f"{metrics_test['rmse']:.3f}",
               f"{metrics_test['mape']:.2f}",
               f"{metrics_test['r2']:.4f}"]
    table = ax.table(
        cellText  = [v_vals, t_vals],
        rowLabels = ["Validation", "Test"],
        colLabels = labels,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.4, 2.2)
    # Colora header e righe
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#d6eaf8")
        elif r == 2:
            cell.set_facecolor("#fde8d8")
        if c == -1:
            cell.set_facecolor("#ecf0f1"); cell.set_text_props(fontweight="bold")
    ax.set_title("Riepilogo Metriche — Validation vs Test", fontsize=13, pad=20)

    plt.suptitle("Rete Neurale Semplice — Risultati", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grafici salvati: {save_path}")

# 8. VALUTAZIONE FINALE E PLOT
y_pred_val = model.predict(X_val).flatten()
y_pred_test = model.predict(X_test).flatten()

metrics_val = evaluate_metrics(y_val, y_pred_val, "Validation")
metrics_test = evaluate_metrics(y_test, y_pred_test, "Test")

plot_results(history, y_val, y_pred_val, y_test, y_pred_test, metrics_val, metrics_test)

print(f"\nR^2 Finale (Test): {metrics_test['r2']:.4f}")
