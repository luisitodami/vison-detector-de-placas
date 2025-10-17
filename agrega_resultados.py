from pathlib import Path
import re
import pandas as pd

# === Ajustes principales ===
RUNS_DIR = Path("runs") / "detect"          # carpeta donde YOLO guarda las corridas
OUT_DIR  = Path("audit_out"); OUT_DIR.mkdir(exist_ok=True)
CSV_OUT  = OUT_DIR / "learning_curve.csv"
XLSX_OUT = OUT_DIR / "learning_curve.xlsx"

# Columnas posibles según versión de YOLOv8
CAND_M50   = ["metrics/mAP50(B)", "val/box/mAP50", "map50"]         # mAP@0.5
CAND_M5095 = ["metrics/mAP50-95(B)", "val/box/mAP50-95", "map"]     # mAP@0.5:0.95
CAND_PREC  = ["metrics/precision(B)", "precision"]                  # Precisión
CAND_REC   = ["metrics/recall(B)", "recall"]                        # Recall
CAND_TPE   = ["time/epoch", "time"]                                 # tiempo por epoch, si aparece

# ===== Helpers =====
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def extract_N_from_name(exp_dir: Path):
    """Primero intenta por nombre del experimento (…_N2000). Si no, intenta leer args/data para train_XXXX."""
    m = re.search(r"_N(\d+)", exp_dir.name)
    if m:
        return int(m.group(1))
    # intenta con args.yaml / opt.yaml / cfg.yaml: buscar 'train_XXXX'
    for fname in ["args.yaml", "opt.yaml", "cfg.yaml"]:
        p = exp_dir / fname
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            m2 = re.search(r"train[_/](\d+)", txt)
            if m2:
                return int(m2.group(1))
    return None

def best_row(df):
    """Devuelve la fila (Series) con mejor mAP@0.5 si existe; si no, la última."""
    c_m50 = pick_col(df, CAND_M50)
    if c_m50 and df[c_m50].notna().any():
        return df.loc[df[c_m50].idxmax()]
    else:
        return df.iloc[-1]

# ===== Recolecta resultados =====
rows = []
if not RUNS_DIR.exists():
    print("No encuentro", RUNS_DIR.resolve())
else:
    for exp_dir in sorted(RUNS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        res = exp_dir / "results.csv"
        if not res.exists():
            continue
        try:
            df = pd.read_csv(res)
        except Exception:
            continue
        if df.empty:
            continue

        N = extract_N_from_name(exp_dir)  # puede quedar None si no reconoce el patrón
        br = best_row(df)

        c_m50   = pick_col(df, CAND_M50)
        c_m5095 = pick_col(df, CAND_M5095)
        c_p     = pick_col(df, CAND_PREC)
        c_r     = pick_col(df, CAND_REC)
        c_tpe   = pick_col(df, CAND_TPE)

        row = {
            "exp": exp_dir.name,
            "N_train": N,
            "epoch": int(br.get("epoch", len(df)-1)),
            "mAP50": float(br.get(c_m50, float("nan"))) if c_m50 else float("nan"),
            "mAP50_95": float(br.get(c_m5095, float("nan"))) if c_m5095 else float("nan"),
            "precision": float(br.get(c_p, float("nan"))) if c_p else float("nan"),
            "recall": float(br.get(c_r, float("nan"))) if c_r else float("nan"),
        }
        # tiempo total aprox si hay time/epoch
        if c_tpe:
            try:
                total_time = float(df[c_tpe].sum())
                row["time_total_epochs(s)"] = total_time
            except Exception:
                pass

        rows.append(row)

# ===== Tabla ordenada por N (si falta, por nombre)
if not rows:
    print("No se encontraron results.csv en", RUNS_DIR.resolve())
    raise SystemExit

tab = pd.DataFrame(rows)
if tab["N_train"].notna().any():
    tab = tab.sort_values(["N_train", "exp"], ascending=[True, True])
else:
    tab = tab.sort_values("exp")

tab.to_csv(CSV_OUT, index=False)
print("CSV guardado:", CSV_OUT)

# ===== Excel + gráfico mAP50 vs N =====
with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as xw:
    tab.to_excel(xw, sheet_name="learning_curve", index=False)

    # gráfico (si tenemos N y mAP50)
    if tab["N_train"].notna().any() and tab["mAP50"].notna().any():
        from openpyxl.chart import LineChart, Reference
        ws = xw.sheets["learning_curve"]
        # ubicar rangos
        n_rows = len(tab) + 1  # + header
        # categorías (N_train)
        cat = Reference(ws, min_col=2, min_row=2, max_row=n_rows)   # col B = N_train
        # serie mAP50
        m50_col = list(tab.columns).index("mAP50") + 1  # 1-indexed
        data = Reference(ws, min_col=m50_col, min_row=1, max_row=n_rows)  # incluye header
        chart = LineChart()
        chart.title = "mAP@0.5 vs N (train)"
        chart.y_axis.title = "mAP@0.5"
        chart.x_axis.title = "N (imágenes de train)"
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cat)
        chart.marker = None
        ws.add_chart(chart, "H2")

print("Excel guardado:", XLSX_OUT)
print("\nTIPO: si ves N_train = NaN, renombra tus corridas como '..._N0500', '..._N1000', etc., o asegúrate de que 'args.yaml' contenga rutas 'train_XXXX'.")
