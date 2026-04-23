"""
╔══════════════════════════════════════════════════════════════════════╗
║          LOAD FORECASTING — CNN-LSTM  (Keras / TensorFlow)          ║
╚══════════════════════════════════════════════════════════════════════╝

DATASET
───────
  • 50.376 righe  =  2.099 giorni × 24 ore
  • TIMESTAMP : ora del giorno (0–23)
  • LOAD       : target continuo [48.4 – 315.6]
  • w1–w25     : 25 sensori interi [0 – 104], alta correlazione reciproca

═══════════════════════════════════════════════════════════════════════
PARTE 1 — SANITIZZAZIONE
═══════════════════════════════════════════════════════════════════════

Obiettivo: eliminare rumore e outlier PRIMA di costruire le finestre
temporali, in modo che il modello non impari pattern spuri.

1. CLIP FISICO SENSORI [0, 104]
   Valori fuori range fisico sono impossibili → azzeramento diretto.

2. SMOOTHING SENSORI — rimozione spike isolati
   Per ogni sensore wk calcoliamo la differenza prima-ordine |Δx_t|.
   Se |Δx_t| > 6 × MAD(|Δx|) il campione è un glitch isolato:
   lo sostituiamo con NaN e poi interpoliamo linearmente tra i vicini.
   Usiamo MAD (Median Absolute Deviation) invece della std perché è
   robusta agli outlier stessi.

3. OUTLIER LOAD PER FASCIA ORARIA — clip percentilico
   Il LOAD ha una distribuzione diversa a ogni ora (es. ore notturne
   hanno media ~115, ore serali ~170). Applicare un unico threshold
   globale sarebbe sbagliato.
   → Per ogni ora h ∈ [0,23]: clip al [1°, 99°] percentile di quella
     fascia. Elimina valori estremi senza distorcere la distribuzione.

4. ROLLING MEDIAN 3pt SUL LOAD
   Sostituisce ogni campione LOAD con la mediana su finestra centrata
   di 3 punti. Rimuove glitch singoli rimasti dopo lo step 3, senza
   introdurre lag (finestra centrata).

═══════════════════════════════════════════════════════════════════════
PARTE 2 — FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════════

1. ENCODING CICLICO DELL'ORA
   L'ora del giorno è periodica: l'ora 23 è "vicina" all'ora 0.
   Un intero 0–23 non cattura questa circolarità.
   Trasformazione:
       hour_sin = sin(2π × h / 24)
       hour_cos = cos(2π × h / 24)
   Le due componenti insieme identificano univocamente ogni ora
   e preservano la metrica circolare.

2. NESSUNA PCA SUI SENSORI
   I sensori sono correlati MA la CNN 1D impara già a combinare le
   feature ridondanti in modo ottimale. La PCA qui sottrae
   interpretabilità senza beneficio reale.

═══════════════════════════════════════════════════════════════════════
PARTE 3 — NORMALIZZAZIONE
═══════════════════════════════════════════════════════════════════════

StandardScaler (z-score) fit SOLO sul training set, poi applicato
identicamente a validation e test → zero data leakage.

LOAD viene normalizzato come le altre feature: il modello lavora in
spazio normalizzato, le metriche finali vengono de-normalizzate.

═══════════════════════════════════════════════════════════════════════
PARTE 4 — FINESTRE TEMPORALI (Sliding Window)
═══════════════════════════════════════════════════════════════════════

Input:  X[i]  = array (SEQ_LEN × n_features)  — le ultime SEQ_LEN ore
Target: y[i]  = LOAD[i + SEQ_LEN]             — ora successiva

SEQ_LEN = 48 ore (2 giorni): cattura il profilo giornaliero completo
più la dipendenza col giorno precedente.

═══════════════════════════════════════════════════════════════════════
PARTE 5 — ARCHITETTURA CNN-LSTM
═══════════════════════════════════════════════════════════════════════

Perché meglio degli alberi su questo problema:

  Gli alberi (LightGBM, XGBoost) operano su FEATURE PUNTALI: vedono
  un vettore di feature al tempo t, non la sequenza. I lag features
  li compensano parzialmente, ma non catturano pattern sottili nella
  forma della curva.

  CNN-LSTM opera sulla SEQUENZA INTERA:
  - La CNN vede la finestra come un segnale 1D e impara filtri locali
    (es. "picco serale", "rampa mattutina")
  - L'LSTM memorizza dipendenze a lungo termine tra giorni

STACK DEL MODELLO:

  Input: (batch, 48, 28)   ← 48 timestep, 28 feature (LOAD+25sensor+sin+cos)
    │
    ├─ Conv1D(64 filtri, kernel=3, padding='same', ReLU)
    │    Impara pattern locali su finestre di 3 ore.
    │    64 filtri = 64 pattern locali diversi cercati.
    │    padding='same' → lunghezza temporale invariata.
    │
    ├─ Conv1D(64 filtri, kernel=3, padding='same', ReLU)
    │    Secondo strato CNN: combina i pattern del primo in pattern
    │    più astratti (es. "rampa + plateau").
    │
    ├─ Dropout(0.2)
    │    Spegne casualmente il 20% dei neuroni → regolarizzazione.
    │
    ├─ LSTM(128 unità, return_sequences=True)
    │    128 celle di memoria. return_sequences=True → passa l'intera
    │    sequenza temporale allo strato LSTM successivo.
    │    L'LSTM cattura: "dopo 2 giorni con picco serale alto, il
    │    giorno successivo tende a…"
    │
    ├─ Dropout(0.2)
    │
    ├─ LSTM(64 unità)
    │    Condensa la sequenza in un vettore fisso di 64 valori
    │    (l'ultimo hidden state riassume tutta la storia).
    │
    ├─ Dense(64, ReLU)
    │    Layer fully-connected: combina le rappresentazioni LSTM.
    │
    ├─ Dropout(0.2)
    │
    └─ Dense(1)              ← output: LOAD normalizzato

Totale parametri: ~180.000 (leggero, veloce su CPU)

LOSS FUNCTION: Huber (δ=1)
  Ibrida tra MSE e MAE: quadratica per errori piccoli (come MSE),
  lineare per errori grandi (robusta agli outlier residui come MAE).

OPTIMIZER: Adam(lr=1e-3) con ReduceLROnPlateau
  Dimezza il learning rate se val_loss non migliora per 4 epoche.

═══════════════════════════════════════════════════════════════════════
PARTE 6 — SPLIT TEMPORALE (NO DATA LEAKAGE)
═══════════════════════════════════════════════════════════════════════

  Train  → giorni   0 – 1.709  (~81%)
  Val    → giorni 1.710 – 1.919 (~9%)   [early stopping + LR decay]
  Test   → giorni 1.920 – 2.099 (~10%)  [valutazione finale]

  MAI shuffle temporale: il futuro non deve entrare nel passato.

"""

# ══════════════════════════════════════════════════════════════════════
# DIPENDENZE
# pip install tensorflow keras scikit-learn pandas numpy matplotlib
# ══════════════════════════════════════════════════════════════════════

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # rimuovi se vuoi finestre interattive
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import keras
from keras import layers, callbacks as kcb

# ─── Iperparametri ────────────────────────────────────────────────────
SEQ_LEN   = 48      # ore di contesto (finestra input)
BATCH     = 256
EPOCHS    = 60
PATIENCE  = 10      # early stopping


# ══════════════════════════════════════════════════════════════════════
# 1. SANITIZZAZIONE
# ══════════════════════════════════════════════════════════════════════

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("w")]

    # ── 1a. Clip fisico sensori ──────────────────────────────────────
    df[sensor_cols] = df[sensor_cols].clip(0, 104)

    # ── 1b. Rimozione spike isolati per ogni sensore ─────────────────
    for col in sensor_cols:
        s    = df[col].astype(float)
        diff = s.diff().abs()
        mad  = diff.median()                        # MAD robusto
        # Un campione è spike se la variazione supera 6×MAD
        spike_mask = diff > 6 * (mad + 1e-6)
        s[spike_mask] = np.nan                      # → NaN
        df[col] = s.interpolate("linear").bfill().ffill()

    # ── 1c. Outlier LOAD per fascia oraria ───────────────────────────
    for h in range(24):
        mask       = df["TIMESTAMP"] == h
        q1, q3     = df.loc[mask, "LOAD"].quantile([0.01, 0.99])
        df.loc[mask, "LOAD"] = df.loc[mask, "LOAD"].clip(q1, q3)

    # ── 1d. Smoothing LOAD con rolling median 3pt ────────────────────
    df["LOAD"] = df["LOAD"].rolling(3, center=True, min_periods=1).median()

    return df


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING + NORMALIZZAZIONE
# ══════════════════════════════════════════════════════════════════════

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["TIMESTAMP"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["TIMESTAMP"] / 24)
    return df


def fit_normalize(df_train, all_cols):
    """Calcola media e std sul solo training set."""
    mu = df_train[all_cols].mean()
    sg = df_train[all_cols].std().replace(0, 1)
    return mu, sg


def apply_normalize(df, mu, sg, all_cols):
    df = df.copy()
    df[all_cols] = (df[all_cols] - mu) / sg
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. COSTRUZIONE FINESTRE TEMPORALI
# ══════════════════════════════════════════════════════════════════════

def make_windows(arr: np.ndarray, seq_len: int, target_col_idx: int = 0):

    X, y = [], []
    # Creiamo una maschera per prendere tutte le colonne TRANNE quella del target
    input_indices = [i for i in range(arr.shape[1]) if i != target_col_idx]
    for i in range(len(arr) - seq_len):
# Prendiamo solo le colonne di input (sensori + tempo)
        X.append(arr[i : i + seq_len, input_indices]) 
        # Prendiamo solo la colonna target per il timestamp futuro
        y.append(arr[i + seq_len, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# 4. ARCHITETTURA CNN-LSTM
# ══════════════════════════════════════════════════════════════════════

def build_cnn_lstm(seq_len: int, n_features: int) -> keras.Model:
    """
    Input  : (batch, seq_len, n_features)
    Output : (batch, 1)  — LOAD normalizzato
    """
    inp = keras.Input(shape=(seq_len, n_features), name="input")

    # Blocco CNN — pattern locali
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Blocco LSTM — dipendenze temporali
    x = layers.LSTM(128, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(0.2, name="drop2")(x)
    x = layers.LSTM(64, name="lstm2")(x)

    # Testa di regressione
    x   = layers.Dense(64, activation="relu", name="dense1")(x)
    x   = layers.Dropout(0.2, name="drop3")(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inp, out, name="CNN_LSTM_LoadForecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(delta=1.0),   # robusta agli outlier
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_model(model, X_tr, y_tr, X_vl, y_vl):
    cbs = [
        kcb.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        kcb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        kcb.ModelCheckpoint(
            "best_load_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=1,
    )
    return history


# ══════════════════════════════════════════════════════════════════════
# 6. VALUTAZIONE
# ══════════════════════════════════════════════════════════════════════

def evaluate(y_true_norm, y_pred_norm, load_mean, load_std, label=""):
    """De-normalizza e stampa le metriche nel dominio originale."""
    y  = y_true_norm * load_std + load_mean
    yp = y_pred_norm * load_std + load_mean

    mae  = mean_absolute_error(y, yp)
    rmse = np.sqrt(mean_squared_error(y, yp))
    r2   = r2_score(y, yp)
    mape = np.mean(np.abs((y - yp) / (np.abs(y) + 1e-8))) * 100

    print(f"\n{'─'*42}")
    print(f"  {label}")
    print(f"{'─'*42}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


# ══════════════════════════════════════════════════════════════════════
# 7. VISUALIZZAZIONE
# ══════════════════════════════════════════════════════════════════════

def plot_results(history,
                 y_val_norm,  yp_val_norm,
                 y_test_norm, yp_test_norm,
                 val_hours, test_hours,
                 metrics_val, metrics_test,
                 load_mean, load_std,
                 save_path="load_deep_results.png"):

    # De-normalizza
    y_v  = y_val_norm   * load_std + load_mean
    yp_v = yp_val_norm  * load_std + load_mean
    y_t  = y_test_norm  * load_std + load_mean
    yp_t = yp_test_norm * load_std + load_mean

    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    # ── [0,0-1] Loss curve ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(history.history["loss"],     label="Train loss", linewidth=1.5)
    ax.plot(history.history["val_loss"], label="Val loss",   linewidth=1.5)
    ax.set_title("Huber Loss — Training / Validation", fontsize=13)
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # ── [1, 0] Predizioni Validation (7 giorni) ───────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n  = min(7 * 24, len(y_v))
    ax.plot(y_v[-n:],  label="Reale",    linewidth=1.2)
    ax.plot(yp_v[-n:], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title("Validation — ultimi 7 giorni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [1, 1] Predizioni Test (7 giorni) ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n  = min(7 * 24, len(y_t))
    ax.plot(y_t[-n:],  label="Reale",    linewidth=1.2)
    ax.plot(yp_t[-n:], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title("Test — ultimi 7 giorni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 0] Scatter Val: Reale vs Predetto ────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(y_v, yp_v, alpha=0.15, s=4, color="steelblue")
    mn, mx = min(y_v.min(), yp_v.min()), max(y_v.max(), yp_v.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_title(f"Validation — Reale vs Predetto  (R²={metrics_val['r2']:.4f})", fontsize=12)
    ax.set_xlabel("LOAD Reale"); ax.set_ylabel("LOAD Predetto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 1] Scatter Test: Reale vs Predetto ───────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(y_t, yp_t, alpha=0.15, s=4, color="darkorange")
    mn, mx = min(y_t.min(), yp_t.min()), max(y_t.max(), yp_t.max())
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

    plt.suptitle("CNN-LSTM Load Forecasting — Risultati", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grafici salvati: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════

def main(data_path: str = "Dataset_trainingSanitificato.csv"):

    print("=" * 55)
    print("  LOAD FORECASTING — CNN-LSTM + Data Sanitization")
    print("=" * 55)

    # ── Carica ────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df["day"] = df.index // 24

    # ── Sanitizza ─────────────────────────────────────────────────────
    print("\n[1] Sanitizzazione dati...")
    df_clean = sanitize(df)
    print(f"  LOAD prima → mean={df['LOAD'].mean():.2f}  std={df['LOAD'].std():.2f}")
    print(f"  LOAD dopo  → mean={df_clean['LOAD'].mean():.2f}  std={df_clean['LOAD'].std():.2f}")

    # ── Feature engineering ───────────────────────────────────────────
    df_clean = add_cyclic_features(df_clean)
    sensor_cols  = [c for c in df.columns if c.startswith("w")]
    feautres_cols = sensor_cols + ["hour_sin", "hour_cos"]
    #2. Definiamo la colonna che vogliamo prevedere (target)
    target_col = ["LOAD"]
    all_cols = feautres_cols + target_col
    # 4. TARGET_IDX ora punta all'ultima colonna della nostra nuova lista
    TARGET_IDX = all_cols.index("LOAD")

    # ── Split temporale 80 / 10 / 10 sui giorni ──────────────────────
    print("\n[2] Split temporale 80/10/10 (no data leakage)...")
    total_days = df_clean["day"].nunique()
    train_end  = int(total_days * 0.80)
    val_end    = int(total_days * 0.90)

    df_train = df_clean[df_clean["day"] <  train_end].copy()
    df_val   = df_clean[(df_clean["day"] >= train_end) & (df_clean["day"] < val_end)].copy()
    df_test  = df_clean[df_clean["day"] >= val_end].copy()
    print(f"  Totale giorni : {total_days}")
    print(f"  Train : {df_train['day'].nunique()} giorni  ({len(df_train)} righe)  [0–{train_end-1}]")
    print(f"  Val   : {df_val['day'].nunique()} giorni  ({len(df_val)} righe)  [{train_end}–{val_end-1}]")
    print(f"  Test  : {df_test['day'].nunique()} giorni  ({len(df_test)} righe)  [{val_end}–{total_days-1}]")

    # ── Normalizzazione (fit solo su train) ───────────────────────────
    print("\n[3] Normalizzazione (z-score, fit su train)...")
    mu, sg        = fit_normalize(df_train, all_cols)
    load_mean     = float(df_train["LOAD"].mean())
    load_std      = float(df_train["LOAD"].std())
    df_train      = apply_normalize(df_train, mu, sg, all_cols)
    df_val        = apply_normalize(df_val,   mu, sg, all_cols)
    df_test       = apply_normalize(df_test,  mu, sg, all_cols)

    # ── Finestre ─────────────────────────────────────────────────────
    print(f"\n[4] Costruzione finestre (SEQ_LEN={SEQ_LEN})...")
    X_tr, y_tr = make_windows(df_train[all_cols].values, SEQ_LEN, TARGET_IDX)
    X_vl, y_vl = make_windows(df_val[all_cols].values,   SEQ_LEN, TARGET_IDX)
    X_te, y_te = make_windows(df_test[all_cols].values,  SEQ_LEN, TARGET_IDX)
    print(f"  X_train={X_tr.shape}  X_val={X_vl.shape}  X_test={X_te.shape}")

    # ── Modello ───────────────────────────────────────────────────────
    model = build_cnn_lstm(SEQ_LEN, len(feautres_cols))
    print(f"\n[5] Modello CNN-LSTM | parametri totali: {model.count_params():,}")
    model.summary()

    # ── Training ─────────────────────────────────────────────────────
    print(f"\n[6] Training (max {EPOCHS} epoche, patience={PATIENCE})...")
    history = train_model(model, X_tr, y_tr, X_vl, y_vl)

    # ── Valutazione ───────────────────────────────────────────────────
    print("\n[7] Valutazione finale...")
    yp_vl = model.predict(X_vl, verbose=0).flatten()
    yp_te = model.predict(X_te, verbose=0).flatten()
    m_val  = evaluate(y_vl, yp_vl, load_mean, load_std, "Validation")
    m_test = evaluate(y_te, yp_te, load_mean, load_std, "Test")

    # ── Plot ─────────────────────────────────────────────────────────
    val_hours  = df_val["TIMESTAMP"].values[SEQ_LEN:]
    test_hours = df_test["TIMESTAMP"].values[SEQ_LEN:]
    plot_results(
        history,
        y_vl,  yp_vl,
        y_te,  yp_te,
        val_hours, test_hours,
        m_val, m_test,
        load_mean, load_std,
    )

    # ── Salva modello ─────────────────────────────────────────────────
    model.save("load_cnn_lstm_final.keras")
    print("  Modello salvato: load_cnn_lstm_final.keras")

    return model, history


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Modifica data_path con il percorso al tuo CSV
    main(data_path="Dataset_trainingSanitificato.csv")
