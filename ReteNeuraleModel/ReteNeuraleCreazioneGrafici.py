import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_metrics(y_true, y_pred, label=""):
    """
    Calcola e stampa le metriche. 
    y_true e y_pred devono essere già nel dominio originale (de-normalizzati).
    """
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
                 save_path="rete_singleton_results.png"):
    """
    Genera i grafici di performance.
    y_val, yp_val, y_test, yp_test devono essere de-normalizzati.
    """
    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    # ── [0,0-1] Loss curve ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(history.history["loss"],     label="Train loss", linewidth=1.5)
    ax.plot(history.history["val_loss"], label="Val loss",   linewidth=1.5)
    ax.set_title("Huber Loss — Training / Validation", fontsize=13)
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # ── [1, 0] Predizioni Validation (Subset) ─────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n  = min(168, len(y_val))
    ax.plot(y_val[:n],  label="Reale",    linewidth=1.2)
    ax.plot(yp_val[:n], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title(f"Validation — Primi {n} campioni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [1, 1] Predizioni Test (Subset) ───────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n  = min(168, len(y_test))
    ax.plot(y_test[:n],  label="Reale",    linewidth=1.2)
    ax.plot(yp_test[:n], label="Predetto", linewidth=1.2, linestyle="--")
    ax.set_title(f"Test — Primi {n} campioni", fontsize=12)
    ax.set_ylabel("LOAD"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 0] Scatter Val ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(y_val, yp_val, alpha=0.15, s=4, color="steelblue")
    mn, mx = min(y_val.min(), yp_val.min()), max(y_val.max(), yp_val.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_title(f"Validation — Reale vs Predetto  (R²={metrics_val['r2']:.4f})", fontsize=12)
    ax.set_xlabel("LOAD Reale"); ax.set_ylabel("LOAD Predetto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2, 1] Scatter Test ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(y_test, yp_test, alpha=0.15, s=4, color="darkorange")
    mn, mx = min(y_test.min(), yp_test.min()), max(y_test.max(), yp_test.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_title(f"Test — Reale vs Predetto  (R²={metrics_test['r2']:.4f})", fontsize=12)
    ax.set_xlabel("LOAD Reale"); ax.set_ylabel("LOAD Predetto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [3, :] Tabella metriche ───────────────────────────────────────
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

    plt.suptitle("Rete Neurale Semplice (Singleton) — Risultati", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grafici salvati: {save_path}")
