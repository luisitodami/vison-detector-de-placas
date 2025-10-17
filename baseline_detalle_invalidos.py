from pathlib import Path
import csv

ROOT=Path("."); SPLITS=["train","valid","test"]
IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
MIN_BOX_AREA=0.0005

def parse(lbl):
    issues=[]; cls=[]
    if not lbl.exists(): return False,["missing"],cls
    try:
        lines=[ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception: return False,["parse_error"],cls
    if not lines: return False,["empty"],cls
    ok_big=False
    for ln in lines:
        parts=ln.split()
        if len(parts)<5: return False,["format_error"],cls
        try:
            c=int(parts[0]); x,y,w,h=map(float,parts[1:5])
        except Exception: return False,["parse_error"],cls
        cls.append(c)
        if not(0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1):
            return False,["coords_out_of_range"],cls
        if w*h>=MIN_BOX_AREA: ok_big=True
    if not ok_big: return False,["only_tiny_boxes"],cls
    return True,[],cls

rows=[]
for split in SPLITS:
    imgs=sorted([p for p in (ROOT/split/"images").glob("*") if p.suffix.lower() in IMG_EXTS])
    for img in imgs:
        lbl=img.parent.parent/"labels"/(img.stem+".txt")
        ok,issues,_=parse(lbl)
        if not ok:
            rows.append([split,str(img.name),",".join(issues),str(lbl.name if lbl.exists() else "N/A")])

out=ROOT/"audit_out"/"baseline_labels_invalidos.csv"
out.parent.mkdir(exist_ok=True)
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["split","imagen","motivo","label_file"])
    w.writerows(rows)
print("Detalle guardado en:", out, f"({len(rows)} archivos)")
