from pathlib import Path

# Ajustes
MIN_BOX_AREA = 0.0005  # descarta cajas minúsculas
SPLITS_TO_FIX = [
    Path("valid/labels"),
    Path("test/labels"),
    # Subsets que vas a entrenar:
    Path("subsets_series/train_500/labels"),
    Path("subsets_series/train_1000/labels"),
    Path("subsets_series/train_1500/labels"),
    Path("subsets_series/train_2000/labels"),
    Path("subsets_series/train_2514/labels"),
]

def clamp01(x):
    return max(0.0, min(1.0, x))

def line_to_bbox(parts):
    """
    parts: lista de strings. Formatos posibles:
    - Detect:  class cx cy w h
    - Segment: class x1 y1 x2 y2 ... (coords normalizadas)
    Devuelve tuple (cls, cx, cy, w, h) o None si no se puede.
    """
    if len(parts) < 5:
        return None
    try:
        cls = 0  # forzamos clase única
        nums = list(map(float, parts[1:]))
    except Exception:
        return None

    if len(nums) == 4:
        # ya está en formato bbox
        cx, cy, w, h = nums
    else:
        # formato segmento: pares (x,y)
        if len(nums) % 2 != 0 or len(nums) < 6:
            return None
        xs = nums[0::2]
        ys = nums[1::2]
        # min/max ya están normalizados 0..1 (asumido)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        # clamp a [0,1] por seguridad
        x1 = clamp01(x1); x2 = clamp01(x2)
        y1 = clamp01(y1); y2 = clamp01(y2)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        cx = x1 + w/2
        cy = y1 + h/2

    # validar rangos
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    if w * h < MIN_BOX_AREA:
        return None
    return (cls, cx, cy, w, h)

def sanitize_dir(lbl_dir: Path):
    fixed_files = 0
    dropped_lines = 0
    total_lines = 0
    changed = 0

    if not lbl_dir.exists():
        print(f"[AVISO] No existe {lbl_dir}")
        return

    for txt in lbl_dir.glob("*.txt"):
        try:
            lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines()]
        except Exception:
            continue
        out = []
        any_change = False
        for ln in lines:
            if not ln:
                continue
            total_lines += 1
            parts = ln.split()
            bbox = line_to_bbox(parts)
            if bbox is None:
                dropped_lines += 1
                any_change = True
                continue
            cls, cx, cy, w, h = bbox
            new_ln = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            if new_ln != ln or parts[0] != "0" or len(parts) != 5:
                any_change = True
            out.append(new_ln)

        if any_change:
            if out:
                txt.write_text("\n".join(out) + "\n", encoding="utf-8")
                fixed_files += 1
                changed += 1
            else:
                # si todas las líneas eran inválidas, deja el archivo vacío
                txt.write_text("", encoding="utf-8")
                changed += 1

    print(f"[OK] {lbl_dir} -> archivos modificados: {fixed_files}, líneas totales: {total_lines}, líneas descartadas: {dropped_lines}, cambiados: {changed}")

def main():
    for d in SPLITS_TO_FIX:
        sanitize_dir(d)

if __name__ == "__main__":
    main()
