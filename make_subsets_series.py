from pathlib import Path
from collections import defaultdict
import shutil, csv
import cv2, numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

ROOT = Path(".")
SRC_IMG = ROOT/"train/images"
SRC_LBL = ROOT/"train/labels"
OUT_ROOT = ROOT/"subsets_series"
OUT_ROOT.mkdir(exist_ok=True, parents=True)
AUDIT = ROOT/"audit_out"; AUDIT.mkdir(exist_ok=True)

# tamaños que vamos a construir (capados por el tamaño real de train)
TARGETS = [500, 1000, 1500, 2000, 2514]

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
MIN_W, MIN_H = 320, 240
MIN_VAR_LAPLACE = 20.0
MIN_BOX_AREA = 0.0005
PHASH_HAMMING_MAX = 3           # dedup intra-train
PHASH_PREFIX = 4

def read_img(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is not None: return img
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if hasattr(cv2, "imdecode"):
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None: return img
    except Exception: pass
    try:
        im = Image.open(p).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def lap_var(img): return float(cv2.Laplacian(img, cv2.CV_64F).var())

def yolo_ok(lbl: Path):
    if not lbl.exists(): return False
    try:
        lines=[ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception: return False
    if not lines: return False
    any_big=False
    for ln in lines:
        s=ln.split()
        if len(s)<5: return False
        try:
            x,y,w,h = map(float, s[1:5])
        except Exception: return False
        if not (0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1): return False
        if w*h >= MIN_BOX_AREA: any_big=True
    return any_big

def phash_hex(p):
    try: return str(imagehash.phash(Image.open(p).convert("RGB")))
    except Exception: return ""

def ham(a,b): return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
def quality_key(w,h,var): return (w*h, var)

# 1) Recolectar candidatos "buenos" del train
items=[]
imgs=[p for p in SRC_IMG.glob("*") if p.suffix.lower() in IMG_EXTS]
for img in tqdm(imgs, desc="Escaneando train"):
    lbl=SRC_LBL/(img.stem+".txt")
    if not yolo_ok(lbl): continue
    bgr=read_img(img)
    if bgr is None: continue
    h,w=bgr.shape[:2]
    if w<MIN_W or h<MIN_H: continue
    var=lap_var(bgr)
    if var<MIN_VAR_LAPLACE: continue
    pha=phash_hex(img)
    items.append({"img":img,"lbl":lbl,"w":w,"h":h,"var":var,"pha":pha})

print("Candidatos tras filtros:", len(items))

# 2) Deduplicación intra-train por pHash (clusters y conservar el de mayor calidad)
buckets=defaultdict(list)
for it in items:
    key = it["pha"][:PHASH_PREFIX] if it["pha"] else f"nohash_{it['img'].suffix}"
    buckets[key].append(it)

visited=set(); pool=[]
for _, lst in tqdm(buckets.items(), desc="Deduplicando pHash"):
    n=len(lst)
    taken=[False]*n
    for i in range(n):
        if taken[i]: continue
        cluster=[lst[i]]; taken[i]=True
        for j in range(i+1,n):
            if taken[j]: continue
            a,b=lst[i], lst[j]
            if a["pha"] and b["pha"] and ham(a["pha"], b["pha"])<=PHASH_HAMMING_MAX:
                cluster.append(b); taken[j]=True
        best=max(cluster, key=lambda x: quality_key(x["w"],x["h"],x["var"]))
        if best["img"] not in visited:
            pool.append(best); visited.add(best["img"])

# añade los que no entraron por hash (raros)
for it in items:
    if it["img"] not in visited:
        pool.append(it); visited.add(it["img"])

# dedup final por ruta
pool = list({it["img"]:it for it in pool}.values())
pool.sort(key=lambda x: quality_key(x["w"],x["h"],x["var"]), reverse=True)
print("Post-dedup:", len(pool))

# 3) Construir subsets cumulativos
report_rows=[]
maxN = len(pool)
targets = [min(t, maxN) for t in TARGETS]
# Garantiza acumulativo: top-N viene del mismo ranking
for N in targets:
    out_dir = OUT_ROOT/f"train_{N}"
    (out_dir/"images").mkdir(parents=True, exist_ok=True)
    (out_dir/"labels").mkdir(parents=True, exist_ok=True)

    subset = pool[:N]
    # copiar
    for it in tqdm(subset, desc=f"Copiando subset {N}"):
        shutil.copy2(it["img"], out_dir/"images"/it["img"].name)
        shutil.copy2(it["lbl"], out_dir/"labels"/it["lbl"].name)

    # lista y yaml
    lst = out_dir/f"train_{N}.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for it in subset:
            f.write(str((out_dir/"images"/it["img"].name).resolve()).replace("\\","/")+"\n")

    yaml = f"""path: {str(out_dir.resolve()).replace('\\','/')}
train: train_{N}.txt
val: ../../valid/images
test: ../../test/images
nc: 1
names: ["license-plate"]
"""
    (out_dir/f"data_{N}.yaml").write_text(yaml, encoding="utf-8")
    report_rows.append([N, len(subset)])

# 4) Reporte simple
with open(AUDIT/"subsets_series_report.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["N","seleccionados"]); w.writerows(report_rows)

print("Listo. Subsets en:", OUT_ROOT)
print("Reporte:", AUDIT/"subsets_series_report.csv")
print("Entrena con, por ejemplo:\n  yolo detect train data=\"subsets_series/train_1000/data_1000.yaml\" model=\"yolov8n.pt\" imgsz=640 epochs=50 batch=16 device=0 project=\"runs\" name=\"placas_v8n_N1000\"")
py -3.13 make_subsets_series.py
