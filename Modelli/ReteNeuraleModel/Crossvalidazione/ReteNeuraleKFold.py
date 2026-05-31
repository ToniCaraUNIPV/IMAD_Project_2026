import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from keras import layers, models, callbacks

# ══════════════════════════════════════════════════════════════════════
# RIPRODUCIBILITÀ
# ══════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
os.environ['PYTHONHASHSEED']      = str(RANDOM_STATE)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # forza CPU
tf.keras.utils.set_random_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════
# IPERPARAMETRI
# ══════════════════════════════════════════════════════════════════════
N_SPLITS  = 5       # numero di fold K-Fold
EPOCHS    = 60
BATCH     = 32
PATIENCE  = 10      # early stopping per fold

# ══════════════════════════════════════════════════════════════════════
# 1. CARICAMENTO E FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
data_path = 'Dataset/Dataset_random.csv'
if not os.path.exists(data_path):
    data_path = '../Dataset/Dataset_random.csv'

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# Sensori: tutte le colonne dalla terza in poi (esclude TIMESTAMP e LOAD)
sensori_cols = df.columns[2:].tolist()

# Encoding ciclico dell'ora
df['sin_time'] = np.sin(2 * np.pi * df['TIMESTAMP'] / 24)
df['cos_time'] = np.cos(2 * np.pi * df['TIMESTAMP'] / 24)

# Feature finali (NO LOAD in input — solo sensori + tempo)
feature_cols = sensori_cols + ['sin_time', 'cos_time']
X = df[feature_cols].values
y = df['LOAD'].values.reshape(-1, 1)

print(f"Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")
print(f"LOAD — min:{y.min():.1f}  max:{y.max():.1f}  mean:{y.mean():.2f}")

# ══════════════════════════════════════════════════════════════════════
# 2. COSTRUZIONE MODELLO
#    Funzione separata: ogni fold ricrea il modello da zero con gli
#    stessi pesi iniziali per confronto equo tra fold.
# ══════════════════════════════════════════════════════════════════════
def build_model(n_features: int) -> tf.keras.Model:
    init = tf.keras.initializers.GlorotUniform(seed=RANDOM_STATE)
    m = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu', kernel_initializer=init),
        layers.Dense(64,  activation='relu', kernel_initializer=init),
        layers.Dense(32,  activation='relu', kernel_initializer=init),
        layers.Dense(1,   kernel_initializer=init),
    ])
    m.compile(
        optimizer='adam',
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae'],
    )
    return m

# ══════════════════════════════════════════════════════════════════════
# 3. METRICHE
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return dict(mae=mae, rmse=rmse, r2=r2, mape=mape)

# ══════════════════════════════════════════════════════════════════════
# 4. K-FOLD CROSS VALIDATION
# ══════════════════════════════════════════════════════════════════════
# Il dataset è già randomizzato → shuffle=False (ordine già casuale)
kf = KFold(n_splits=N_SPLITS, shuffle=False)

# Strutture per raccogliere i risultati di ogni fold
fold_metrics_val  = []   # lista di dict, una per fold
fold_metrics_test = []
fold_histories    = []   # history Keras per ogni fold
fold_predictions  = []   # (y_true_test, y_pred_test) per scatter

print(f"\n{'='*55}")
print(f"  K-FOLD CROSS VALIDATION  (K={N_SPLITS})")
print(f"{'='*55}")

for fold, (train_val_idx, test_idx) in enumerate(kf.split(X), start=1):

    print(f"\n─── Fold {fold}/{N_SPLITS} ───────────────────────────────────")

    # ── Split: train+val | test ────────────────────────────────────
    X_trainval, X_test_f = X[train_val_idx], X[test_idx]
    y_trainval, y_test_f = y[train_val_idx], y[test_idx]

    # Ulteriore split train | val (15% del train+val totale)
    val_size   = int(len(X_trainval) * 0.15)
    X_train_f  = X_trainval[:-val_size]
    X_val_f    = X_trainval[-val_size:]
    y_train_f  = y_trainval[:-val_size]
    y_val_f    = y_trainval[-val_size:]

    print(f"  Train:{len(X_train_f)}  Val:{len(X_val_f)}  Test:{len(X_test_f)}")

    # ── Normalizzazione (fit SOLO su train del fold) ───────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train_f)
    X_val_s   = scaler_X.transform(X_val_f)
    X_test_s  = scaler_X.transform(X_test_f)

    y_train_s = scaler_y.fit_transform(y_train_f)
    y_val_s   = scaler_y.transform(y_val_f)

    # ── Modello fresco per ogni fold ───────────────────────────────
    tf.keras.utils.set_random_seed(RANDOM_STATE + fold)
    model = build_model(X.shape[1])

    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=1,
    )

    # ── Predizioni e de-normalizzazione ───────────────────────────
    yp_val_s  = model.predict(X_val_s,  verbose=0)
    yp_test_s = model.predict(X_test_s, verbose=0)

    yp_val  = scaler_y.inverse_transform(yp_val_s).flatten()
    yp_test = scaler_y.inverse_transform(yp_test_s).flatten()
    y_val_o  = y_val_f.flatten()
    y_test_o = y_test_f.flatten()

    # ── Metriche ──────────────────────────────────────────────────
    mv = compute_metrics(y_val_o,  yp_val)
    mt = compute_metrics(y_test_o, yp_test)
    fold_metrics_val.append(mv)
    fold_metrics_test.append(mt)
    fold_histories.append(history)
    fold_predictions.append((y_test_o, yp_test))

    print(f"  VAL  → MAE:{mv['mae']:.3f}  RMSE:{mv['rmse']:.3f}  "
          f"MAPE:{mv['mape']:.2f}%  R²:{mv['r2']:.4f}")
    print(f"  TEST → MAE:{mt['mae']:.3f}  RMSE:{mt['rmse']:.3f}  "
          f"MAPE:{mt['mape']:.2f}%  R²:{mt['r2']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 5. RIEPILOGO NUMERICO
# ══════════════════════════════════════════════════════════════════════
def summarize(metrics_list: list, label: str):
    keys = ['mae', 'rmse', 'mape', 'r2']
    print(f"\n{'─'*50}")
    print(f"  {label} — Media ± Std su {N_SPLITS} fold")
    print(f"{'─'*50}")
    for k in keys:
        vals = [m[k] for m in metrics_list]
        print(f"  {k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

summarize(fold_metrics_val,  "VALIDATION")
summarize(fold_metrics_test, "TEST")

# ══════════════════════════════════════════════════════════════════════
# 6. GRAFICI
# ══════════════════════════════════════════════════════════════════════
save_dir  = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(save_dir, "kfold_results.png")

COLORS_VAL  = "#2980b9"
COLORS_TEST = "#e67e22"
FOLD_COLORS = ["#3498db","#e74c3c","#2ecc71","#9b59b6","#f39c12"]

metric_keys   = ['mae', 'rmse', 'mape', 'r2']
metric_labels = ['MAE', 'RMSE', 'MAPE (%)', 'R²']

fig = plt.figure(figsize=(20, 26))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── [0] Boxplot metriche — Validation vs Test ─────────────────────────
ax = fig.add_subplot(gs[0, :])
n_metrics = len(metric_keys)
x         = np.arange(n_metrics)
width     = 0.30

val_data  = [[m[k] for m in fold_metrics_val]  for k in metric_keys]
test_data = [[m[k] for m in fold_metrics_test] for k in metric_keys]

bp_val  = ax.boxplot(val_data,  positions=x - width/2, widths=width,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
bp_test = ax.boxplot(test_data, positions=x + width/2, widths=width,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))

for patch in bp_val['boxes']:
    patch.set_facecolor(COLORS_VAL);  patch.set_alpha(0.7)
for patch in bp_test['boxes']:
    patch.set_facecolor(COLORS_TEST); patch.set_alpha(0.7)

# Sovrapponi i punti dei singoli fold
for i, (vd, td) in enumerate(zip(val_data, test_data)):
    ax.scatter(np.full(N_SPLITS, x[i] - width/2) + np.random.uniform(-0.04,0.04,N_SPLITS),
               vd, color=COLORS_VAL,  s=40, zorder=5, alpha=0.9)
    ax.scatter(np.full(N_SPLITS, x[i] + width/2) + np.random.uniform(-0.04,0.04,N_SPLITS),
               td, color=COLORS_TEST, s=40, zorder=5, alpha=0.9)

ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=12)
ax.legend([bp_val['boxes'][0], bp_test['boxes'][0]],
          ['Validation', 'Test'], fontsize=11)
ax.set_title(f"Distribuzione Metriche per Fold  (K={N_SPLITS})", fontsize=13)
ax.grid(alpha=0.3, axis='y')

# ── [1, 0-1] R² per fold — bar chart affiancato ───────────────────────
ax = fig.add_subplot(gs[1, :])
fold_ids  = np.arange(1, N_SPLITS + 1)
r2_val  = [m['r2'] for m in fold_metrics_val]
r2_test = [m['r2'] for m in fold_metrics_test]
w = 0.35
bars_v = ax.bar(fold_ids - w/2, r2_val,  width=w, color=COLORS_VAL,
                alpha=0.8, label='Validation')
bars_t = ax.bar(fold_ids + w/2, r2_test, width=w, color=COLORS_TEST,
                alpha=0.8, label='Test')
for bar in list(bars_v) + list(bars_t):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
ax.axhline(np.mean(r2_val),  color=COLORS_VAL,  linestyle='--',
           linewidth=1.5, label=f'Media Val R²={np.mean(r2_val):.3f}')
ax.axhline(np.mean(r2_test), color=COLORS_TEST, linestyle='--',
           linewidth=1.5, label=f'Media Test R²={np.mean(r2_test):.3f}')
ax.set_xticks(fold_ids); ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
ax.set_ylabel('R²'); ax.set_ylim(0, 1.05)
ax.set_title('R² per Fold — Validation vs Test', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')

# ── [2, 0] Learning curves medie ──────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
max_ep = max(len(h.history['loss']) for h in fold_histories)
# Allinea le curve (fold con early stop hanno meno epoche → padding con nan)
def pad(arr, length):
    out = np.full(length, np.nan)
    out[:len(arr)] = arr
    return out

train_curves = np.array([pad(h.history['loss'],     max_ep) for h in fold_histories])
val_curves   = np.array([pad(h.history['val_loss'], max_ep) for h in fold_histories])
ep = np.arange(1, max_ep + 1)

for i, (tc, vc) in enumerate(zip(train_curves, val_curves)):
    ax.plot(ep, tc, color=FOLD_COLORS[i], alpha=0.25, linewidth=1)
    ax.plot(ep, vc, color=FOLD_COLORS[i], alpha=0.25, linewidth=1, linestyle='--')

# Media tra fold (ignora nan)
mean_train = np.nanmean(train_curves, axis=0)
mean_val   = np.nanmean(val_curves,   axis=0)
ax.plot(ep, mean_train, color='black',      linewidth=2.0, label='Train (media)')
ax.plot(ep, mean_val,   color='darkorange', linewidth=2.0,
        linestyle='--', label='Val (media)')
ax.set_title('Learning Curves — Media ± Fold singoli', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Huber Loss')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# ── [2, 1] MAE e RMSE per fold — linea ────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
mae_test  = [m['mae']  for m in fold_metrics_test]
rmse_test = [m['rmse'] for m in fold_metrics_test]
ax.plot(fold_ids, mae_test,  marker='o', color=COLORS_VAL,
        linewidth=2, markersize=7, label='MAE Test')
ax.plot(fold_ids, rmse_test, marker='s', color=COLORS_TEST,
        linewidth=2, markersize=7, label='RMSE Test')
ax.axhline(np.mean(mae_test),  color=COLORS_VAL,  linestyle=':', linewidth=1.5)
ax.axhline(np.mean(rmse_test), color=COLORS_TEST, linestyle=':', linewidth=1.5)
ax.set_xticks(fold_ids); ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
ax.set_title('MAE e RMSE per Fold (Test)', fontsize=12)
ax.set_ylabel('Errore'); ax.legend(fontsize=10); ax.grid(alpha=0.3)

# ── [3, 0-1] Scatter Reale vs Predetto per ogni fold ─────────────────
# Disponiamo i 5 scatter in due righe: 3 sopra, 2 sotto
# Usiamo una sub-GridSpec dedicata
gs_scatter = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs[3:5, :], hspace=0.45, wspace=0.35
)
for i, (y_true, y_pred) in enumerate(fold_predictions):
    row, col = divmod(i, 3)
    ax = fig.add_subplot(gs_scatter[row, col])
    mt = fold_metrics_test[i]

    ax.scatter(y_true, y_pred, alpha=0.15, s=5,
               color=FOLD_COLORS[i], rasterized=True)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.2, label='y = x')

    ax.set_title(
        f"Fold {i+1}  —  R²={mt['r2']:.4f}  MAE={mt['mae']:.2f}",
        fontsize=11
    )
    ax.set_xlabel('LOAD Reale', fontsize=9)
    ax.set_ylabel('LOAD Predetto', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Nasconde il sesto riquadro (scatter grid 2×3 ma fold=5)
if N_SPLITS < 6:
    ax_empty = fig.add_subplot(gs_scatter[1, 2])
    ax_empty.axis('off')

# ── Titolo globale e tabella riassuntiva ──────────────────────────────
fig.suptitle(
    f"K-Fold Cross Validation (K={N_SPLITS}) — Rete Neurale LOAD Forecasting",
    fontsize=16, fontweight='bold', y=1.005
)

plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  Grafici salvati: {save_path}")

# ══════════════════════════════════════════════════════════════════════
# 7. STAMPA TABELLA FINALE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print(f"  RIEPILOGO FINALE — {N_SPLITS} FOLD")
print(f"{'═'*55}")
header = f"  {'Fold':>5}  {'MAE_val':>8}  {'R²_val':>7}  {'MAE_tst':>8}  {'R²_tst':>7}"
print(header)
print(f"  {'─'*53}")
for i, (mv, mt) in enumerate(zip(fold_metrics_val, fold_metrics_test), 1):
    print(f"  {i:>5}  {mv['mae']:>8.3f}  {mv['r2']:>7.4f}  "
          f"{mt['mae']:>8.3f}  {mt['r2']:>7.4f}")
print(f"  {'─'*53}")
mv_all = fold_metrics_val;  mt_all = fold_metrics_test
print(f"  {'MEDIA':>5}  "
      f"{np.mean([m['mae'] for m in mv_all]):>8.3f}  "
      f"{np.mean([m['r2']  for m in mv_all]):>7.4f}  "
      f"{np.mean([m['mae'] for m in mt_all]):>8.3f}  "
      f"{np.mean([m['r2']  for m in mt_all]):>7.4f}")
print(f"  {'STD':>5}  "
      f"{np.std([m['mae'] for m in mv_all]):>8.3f}  "
      f"{np.std([m['r2']  for m in mv_all]):>7.4f}  "
      f"{np.std([m['mae'] for m in mt_all]):>8.3f}  "
      f"{np.std([m['r2']  for m in mt_all]):>7.4f}")
print(f"{'═'*55}")
