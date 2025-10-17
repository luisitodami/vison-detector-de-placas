# conteo_post.py
from pathlib import Path
import csv

ROOT = Path(".")
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(p: Path):
    return [x for x in p.glob("*") if x.suffix.lower() in IMG_EXTS]

def contar_split(split: str):
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    imgs = list_images(img_dir)
    lbls = list(lbl_dir.glob("*.txt"))
    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}
    return {
        "split": split,
        "imagenes_total": len(imgs),
        "labels_total": len(lbls),
        "imgs_sin_label": len(img_stems - lbl_stems),
        "labels_sin_img": len(lbl_stems - img_stems),
    }

def main():
    rows = [contar_split(s) for s in SPLITS]
    # imprimir tabla
    hdr = ["split","imagenes_total","labels_total","imgs_sin_label","labels_sin_img"]
    colw = [max(len(str(r[k])) for r in rows + [{k:k}]) for k in hdr]
    def fmt(r): return " | ".join(str(r[k]).ljust(colw[i]) for i,k in enumerate(hdr))
    print("\n=== CONTEO POST-LIMPIEZA ===")
    print(" | ".join(h.ljust(colw[i]) for i,h in enumerate(hdr)))
    print("-+-".join("-"*w for w in colw))
    for r in rows: print(fmt(r))
    print("-+-".join("-"*w for w in colw))
    print("TOTAL imágenes:", sum(r["imagenes_total"] for r in rows),
          "| TOTAL labels:", sum(r["labels_total"] for r in rows))

    # guardar CSV
    out = ROOT/"audit_out"/"resumen_postlimpieza.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows: w.writerow(r)
    print("\nResumen guardado en:", out)

if __name__ == "__main__":
    main()
