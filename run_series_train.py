# run_series_train.py
import subprocess, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== CONFIG ====================
NS = [500, 1000, 1500, 2000, 2514]   # tamaños de train a correr
MODEL = "yolov8n.pt"
IMGSZ = 640
EPOCHS = 50

# FORZAR CPU SIEMPRE (sin autodetección)
DEVICE = "cpu"
BATCH  = 8

PROJECT = "runs"
NAME_PREFIX = "placas_v8n_N"  # quedará placas_v8n_N0500, etc.

BASE = Path(".")
PLOTS_DIR = BASE/"audit_out"/"plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = BASE/"audit_out"/"learning_curve_incremental.csv"

# Columnas candidatas (cambian entre versiones de Ultralytics)
CAND_M50   = ["metrics/mAP50(B)", "val/box/mAP50", "map50"]
CAND_M5095 = ["metrics/mAP50-95(B)", "val/box/mAP50-95", "map"]
CAND_PREC  = ["metrics/precision(B)", "precision"]
CAND_REC   = ["metrics/recall(B)", "recall"]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    # No detenemos toda la serie si una corrida falla
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    if res.returncode != 0:
        print(f"[AVISO] Falló: {' '.join(cmd)}  — sigo con el siguiente N.")

def plot_and_append(results_csv: Path, exp_name: str, N: int):
    if not results_csv.exists():
        print(f"[AVISO] No encuentro {results_csv}; no genero gráficos para {exp_name}.")
        return

    try:
        df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"[AVISO] No pude leer {results_csv}: {e}")
        return
    if df.empty:
        print(f"[AVISO] {results_csv} está vacío.")
        return

    epoch = df["epoch"] if "epoch" in df.columns else pd.RangeIndex(len(df))
    m50   = df[pick_col(df, CAND_M50)] if pick_col(df, CAND_M50) else None
    m5095 = df[pick_col(df, CAND_M5095)] if pick_col(df, CAND_M5095) else None
    prec  = df[pick_col(df, CAND_PREC)] if pick_col(df, CAND_PREC) else None
    rec   = df[pick_col(df, CAND_REC)]  if pick_col(df, CAND_REC)  else None

    # mAP curves
    plt.figure()
    if m50 is not None:   plt.plot(epoch, m50,   label="mAP@0.5")
    if m5095 is not None: plt.plot(epoch, m5095, label="mAP@0.5:0.95")
    plt.title(f"{exp_name} (N={N})"); plt.xlabel("epoch"); plt.ylabel("mAP")
    plt.legend(); plt.tight_layout()
    out_map = PLOTS_DIR/f"{exp_name}_map.png"
    plt.savefig(out_map, dpi=150); plt.close()

    # Precision / Recall curves
    out_pr = ""
    if (prec is not None) or (rec is not None):
        plt.figure()
        if prec is not None: plt.plot(epoch, prec, label="Precision")
        if rec  is not None: plt.plot(epoch, rec,  label="Recall")
        plt.title(f"{exp_name} (N={N})"); plt.xlabel("epoch"); plt.ylabel("score")
        plt.legend(); plt.tight_layout()
        out_pr = PLOTS_DIR/f"{exp_name}_pr.png"
        plt.savefig(out_pr, dpi=150); plt.close()

    # Mejor época por mAP50 si existe, si no la última
    if m50 is not None and m50.notna().any():
        best_idx = m50.idxmax()
    else:
        best_idx = len(df) - 1
    best = df.iloc[best_idx]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exp": exp_name,
        "N_train": N,
        "best_epoch": int(best.get("epoch", best_idx)),
        "mAP50": float(best.get(pick_col(df, CAND_M50), float("nan"))),
        "mAP50_95": float(best.get(pick_col(df, CAND_M5095), float("nan"))),
        "precision": float(best.get(pick_col(df, CAND_PREC), float("nan"))),
        "recall": float(best.get(pick_col(df, CAND_REC), float("nan"))),
        "results_csv": str(results_csv),
        "plot_map": str(out_map),
        "plot_pr": str(out_pr) if out_pr else ""
    }

    if SUMMARY_CSV.exists():
        pd.concat([pd.read_csv(SUMMARY_CSV), pd.DataFrame([row])], ignore_index=True)\
          .to_csv(SUMMARY_CSV, index=False)
    else:
        pd.DataFrame([row]).to_csv(SUMMARY_CSV, index=False)

    print("Gráficos:", out_map, ("| " + out_pr if out_pr else ""))
    print("Resumen actualizado:", SUMMARY_CSV)

def main():
    for N in NS:
        data_yaml = BASE/"subsets_series"/f"train_{N}"/f"data_{N}.yaml"
        if not data_yaml.exists():
            print(f"[AVISO] No existe {data_yaml}. ¿Ya generaste subsets_series para N={N}? Me salto.")
            continue

        exp_name = f"{NAME_PREFIX}{N:04d}"
        cmd = [
            "yolo", "detect", "train",
            f"data={str(data_yaml)}",
            f"model={MODEL}",
            f"imgsz={IMGSZ}",
            f"epochs={EPOCHS}",
            f"batch={BATCH}",
            f"device={DEVICE}",
            f"project={PROJECT}",
            f"name={exp_name}",
        ]
        run_cmd(cmd)

        # results.csv (ruta estándar)
        results_csv = BASE/PROJECT/"detect"/exp_name/"results.csv"
        if not results_csv.exists():
            # por si la versión guardó en otra subcarpeta
            found = list((BASE/PROJECT).rglob(f"{exp_name}/results.csv"))
            if found:
                results_csv = found[0]

        plot_and_append(results_csv, exp_name, N)

    print("\n=== LISTO ===")
    print("Gráficas por corrida en:", PLOTS_DIR)
    print("Resumen incremental:", SUMMARY_CSV)
    print("Tip: luego puedes correr un agregador para un .xlsx con la curva final.")

if __name__ == "__main__":
    main()
