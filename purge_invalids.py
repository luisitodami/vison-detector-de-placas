from pathlib import Path
import csv, shutil

ROOT = Path(".")
CSV_PATH = ROOT / "audit_out" / "baseline_labels_invalidos.csv"
TRASH = ROOT / "_trash"

def main():
    if not CSV_PATH.exists():
        print("No encuentro:", CSV_PATH)
        print("Primero corre: python baseline_antes.py")
        return

    moved_imgs = moved_lbls = 0
    TRASH.mkdir(exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            split = row.get("split", "").strip()
            img_name = row.get("imagen", "").strip()
            lbl_name = row.get("label_file", "").strip()

            # rutas origen
            img = ROOT / split / "images" / img_name
            lbl = ROOT / split / "labels" / lbl_name if lbl_name and lbl_name != "N/A" else None

            # rutas destino (papelera)
            (TRASH / split / "images").mkdir(parents=True, exist_ok=True)
            (TRASH / split / "labels").mkdir(parents=True, exist_ok=True)

            # mover imagen
            if img.exists():
                shutil.move(str(img), str(TRASH / split / "images" / img.name))
                moved_imgs += 1
            # mover label (si existe)
            if lbl and lbl.exists():
                shutil.move(str(lbl), str(TRASH / split / "labels" / lbl.name))
                moved_lbls += 1

    print(f"Movidas a papelera: {moved_imgs} imágenes y {moved_lbls} labels.")
    print("Papelera en:", TRASH)
    print("Si quieres revertir, solo vuelve a mover los archivos a train/valid/test.")

if __name__ == "__main__":
    main()
