from pathlib import Path
from collections import Counter, defaultdict
import csv, sys, re

# =============== CONFIG ===============
ROOT = Path(".")  # ejecuta este script desde la carpeta raíz del dataset
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Si quieres que compruebe las clases contra data.yaml (opcional):
DATA_YAML = None  # p. ej. Path("data.yaml") o Path("dataset_placas/data_all.yaml")
# Umbral orientativo para "caja diminuta"
MIN_BOX_AREA = 0.0005

# (Opcional) Anota el origen del dataset para el reporte de texto:
ORIGEN_DATASET = "Conjuntos unificados desde Roboflow (Perú), exporte YOLO."

# =============== HELPERS ===============
def list_images(dir_path: Path):
    return [p for p in dir_path.glob("*") if p.suffix.lower() in IMG_EXTS]

def parse_label_file(lbl_path: Path):
    """
    Devuelve: (ok, issues:list[str], class_ids:list[int])
    Valida formato YOLO: 'cls x y w h' y coords en [0,1].
    """
    issues = []
    class_ids = []
    if not lbl_path.exists():
        issues.append("missing")
        return False, issues, class_ids
    try:
        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        issues.append("parse_error")
        return False, issues, class_ids
    if not lines:
        issues.append("empty")
        return False, issues, class_ids

    ok_any_box = False
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            issues.append("format_error")
            return False, issues, class_ids
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            issues.append("parse_error")
            return False, issues, class_ids
        class_ids.append(cls)
        # Coordenadas normalizadas 0..1
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            issues.append("coords_out_of_range")
            return False, issues, class_ids
        if w * h >= MIN_BOX_AREA:
            ok_any_box = True
    if not ok_any_box:
        issues.append("only_tiny_boxes")
        # sigue siendo "no ok" para entrenamiento, pero formato válido
        return False, issues, class_ids

    return True, issues, class_ids

def try_read_data_yaml_names(yaml_path: Path):
    """Intento simple de extraer 'names: [...]' y 'nc:' sin depender de PyYAML."""
    if not yaml_path or not yaml_path.exists():
        return None, None
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    # Busca nc:
    m_nc = re.search(r"\bnc\s*:\s*(\d+)", text)
    nc = int(m_nc.group(1)) if m_nc else None
    # Busca names: [ ... ]
    m_names = re.search(r"names\s*:\s*\[([^\]]*)\]", text)
    names = None
    if m_names:
        raw = m_names.group(1)
        # divide por comas, quita comillas y espacios
        parts = [re.sub(r'["\']', "", s).strip() for s in raw.split(",")]
        names = [p for p in parts if p != ""]
    return nc, names

# =============== MAIN ===============
def main():
    out_dir = ROOT / "audit_out"
    out_dir.mkdir(exist_ok=True)

    # (Opcional) lee data.yaml para validar clases
    allowed_classes = None
    class_names = None
    if DATA_YAML:
        nc, names = try_read_data_yaml_names(DATA_YAML)
        if nc is not None:
            allowed_classes = set(range(nc))
        if names:
            class_names = names

    per_split_rows = []
    class_hist_split = {s: Counter() for s in SPLITS}
    issues_hist_split = {s: Counter() for s in SPLITS}

    total_images_initial = 0
    total_labels_initial = 0

    for split in SPLITS:
        img_dir = ROOT / split / "images"
        lbl_dir = ROOT / split / "labels"
        imgs = list_images(img_dir)
        lbls = list(lbl_dir.glob("*.txt"))

        total_images_initial += len(imgs)
        total_labels_initial += len(lbls)

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}

        imgs_sin_label = sorted(img_stems - lbl_stems)
        labels_sin_img = sorted(lbl_stems - img_stems)

        ok_labels = 0
        invalid_labels = 0

        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            ok, issues, cls_ids = parse_label_file(lbl)
            if not lbl.exists():
                issues_hist_split[split]["label_missing"] += 1
            else:
                # registra issues por tipo
                for it in issues:
                    issues_hist_split[split][it] += 1
            if ok:
                ok_labels += 1
            else:
                invalid_labels += 1

            # conteo de clases vistas
            for c in cls_ids:
                class_hist_split[split][c] += 1
                if allowed_classes is not None and c not in allowed_classes:
                    issues_hist_split[split]["class_out_of_range"] += 1

        per_split_rows.append({
            "split": split,
            "imagenes_total": len(imgs),
            "labels_total": len(lbls),
            "imgs_sin_label": len(imgs_sin_label),
            "labels_sin_img": len(labels_sin_img),
            "labels_ok": ok_labels,
            "labels_invalidos": invalid_labels,
        })

    # ---- imprime tabla “Antes” en consola ----
    hdr = ["split","imagenes_total","labels_total","imgs_sin_label","labels_sin_img","labels_ok","labels_invalidos"]
    colw = [max(len(str(r[k])) for r in per_split_rows + [{k:k}]) for k in hdr]
    def fmt_row(r): return " | ".join(str(r[k]).ljust(colw[i]) for i,k in enumerate(hdr))
    print("\n=== TABLA ANTES (conteo por split) ===")
    print(" | ".join(h.ljust(colw[i]) for i,h in enumerate(hdr)))
    print("-+-".join("-"*w for w in colw))
    for r in per_split_rows:
        print(fmt_row(r))
    print("-+-".join("-"*w for w in colw))
    print("TOTAL imágenes:", total_images_initial, " | TOTAL labels:", total_labels_initial)

    # ---- guarda CSV de la tabla “Antes” ----
    with open(out_dir/"baseline_antes_por_split.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in per_split_rows:
            w.writerow(r)

    # ---- CSV de issues por split ----
    with open(out_dir/"baseline_issues_por_split.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","issue","conteo"])
        for split, cnt in issues_hist_split.items():
            for issue, n in sorted(cnt.items(), key=lambda x:(x[0])):
                w.writerow([split, issue, n])

    # ---- CSV de distribución de clases por split ----
    with open(out_dir/"baseline_clases_por_split.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        # encabezados dinámicos por clase
        # primero reúne todas las clases vistas
        all_classes = sorted(set(c for s in SPLITS for c in class_hist_split[s].keys()))
        head = ["split"] + [f"class_{c}" for c in all_classes]
        w.writerow(head)
        for split in SPLITS:
            row = [split] + [class_hist_split[split].get(c, 0) for c in all_classes]
            w.writerow(row)

    # ---- TXT con resumen y origen del dataset ----
    with open(out_dir/"baseline_resumen.txt","w",encoding="utf-8") as f:
        f.write("== Línea de base (antes de limpiar) ==\n")
        f.write(f"Origen del dataset: {ORIGEN_DATASET}\n\n")
        for r in per_split_rows:
            f.write(f"{r['split']}: imgs={r['imagenes_total']}  labels={r['labels_total']}  imgs_sin_label={r['imgs_sin_label']}  labels_sin_img={r['labels_sin_img']}  labels_ok={r['labels_ok']}  labels_invalidos={r['labels_invalidos']}\n")
        f.write(f"\nTOTAL imágenes: {total_images_initial}  |  TOTAL labels: {total_labels_initial}\n")
        if class_names:
            f.write("\nClases (data.yaml):\n")
            for i,name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")

    print("\nArchivos generados en:", out_dir)
    print(" - baseline_antes_por_split.csv")
    print(" - baseline_issues_por_split.csv")
    print(" - baseline_clases_por_split.csv")
    print(" - baseline_resumen.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
