# limpieza_etapas.py
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, hashlib, shutil
import cv2, numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# ====== CONFIG ======
ROOT = Path(".")
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

# Umbrales
NEAR_DUP_HAMMING = 3      # pHash más estricto (≤3)
MIN_W, MIN_H = 320, 240   # muy pequeñas
MIN_VAR_LAPLACE = 30.0    # borrosas
MIN_BRIGHT, MAX_BRIGHT = 40, 230  # exposición
ALLOWED_CLASSES = None    # None = no forzar clase; {0} si solo “placa”
MIN_BOX_AREA = 0.0005     # cajas diminutas

# Salidas
Q = ROOT/"_quarantine"
LOG_DIR = ROOT/"audit_out"
for d in ["duplicates_exact","duplicates_near","too_small","blurry","exposure_review","bad_label","tiny_box"]:
    (Q/d).mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ====== helpers ======
def sha1(p:Path, chunk=1<<20):
    h=hashlib.sha1()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def phash(p:Path):
    try:
        im = Image.open(p).convert("RGB")
        return imagehash.phash(im)
    except Exception:
        return None

def read_img(p:Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def lap_var(img): return cv2.Laplacian(img, cv2.CV_64F).var()
def bright_v(img): return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[...,2]))

def yolo_read(lbl:Path):
    if not lbl.exists(): return None
    try:
        lines=[ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return None
    out=[]
    for ln in lines:
        s=ln.split()
        if len(s)<5: return None
        try:
            cls=int(s[0]); x,y,w,h=map(float, s[1:5])
        except Exception:
            return None
        out.append((cls,x,y,w,h))
    return out

def yolo_valid(lines):
    if lines is None or len(lines)==0: return False, "label_missing_or_empty"
    tiny=True
    for (cls,x,y,w,h) in lines:
        if ALLOWED_CLASSES is not None and cls not in ALLOWED_CLASSES:
            return False, "class_out_of_range"
        if not (0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1):
            return False, "coords_out_of_range"
        if w*h >= MIN_BOX_AREA: tiny=False
    if tiny: return False, "tiny_box_only"
    return True, ""

def move_pair(img:Path, lbl:Path, dst_dir:Path, reason:str, writer, dry=False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dry:
        writer.writerow([reason, str(img), str(lbl if lbl and lbl.exists() else ""), str(dst_dir), "DRY"])
        return
    dst_img = dst_dir/img.name
    writer.writerow([reason, str(img), str(lbl if lbl and lbl.exists() else ""), str(dst_dir), "MOVED"])
    shutil.move(str(img), str(dst_img))
    if lbl and lbl.exists():
        shutil.move(str(lbl), str(dst_dir/lbl.name))

def scan():
    items=[]
    for split in SPLITS:
        img_dir = ROOT/split/"images"
        lbl_dir = ROOT/split/"labels"
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
        for img in tqdm(imgs, desc=f"escaneando {split}"):
            lbl = lbl_dir/(img.stem+".txt")
            rec = {"split":split, "img":img, "lbl":lbl, "exists":img.exists()}
            bgr = read_img(img)
            if bgr is None:
                rec.update({"w":0,"h":0,"var":0.0,"bri":0.0})
            else:
                h,w = bgr.shape[:2]
                rec.update({"w":w,"h":h,"var":lap_var(bgr),"bri":bright_v(bgr)})
            items.append(rec)
    return items

# ====== Etapas ======
def etapa_A_duplicados_exactos(items, writer, dry):
    bysha=defaultdict(list)
    for it in items:
        if it["img"].exists():
            bysha[sha1(it["img"])].append(it)
    moved=0
    for sha, group in bysha.items():
        if len(group)<=1: continue
        keep = next((g for g in group if g["split"]=="train"), group[0])
        for it in group:
            if it is keep: continue
            if it["img"].exists():
                move_pair(it["img"], it["lbl"], Q/"duplicates_exact", "duplicate_exact", writer, dry)
                moved += 1
    print(f"[A] Duplicados exactos movidos: {moved}")
    return moved

def etapa_B_casi_duplicados(items, writer, dry):
    # pHash solo de archivos existentes
    recs=[]
    for it in items:
        if not it["img"].exists(): continue
        h = phash(it["img"])
        recs.append({**it, "pha": str(h) if h else ""})

    # SOLO entre splits distintos (evita vaciar train)
    buckets=defaultdict(list)
    for r in recs:
        if not r["pha"]: continue
        for other in ("train","valid","test"):
            if other == r["split"]: continue
            key = (r["pha"][:4], tuple(sorted([r["split"], other])))
            buckets[key].append(r)

    moved=0; visited=set()
    def ham(a,b): return imagehash.hex_to_hash(a)-imagehash.hex_to_hash(b)

    for _, grp in buckets.items():
        n=len(grp)
        for i in range(n):
            for j in range(i+1,n):
                a,b = grp[i], grp[j]
                if not a["img"].exists() or not b["img"].exists(): continue
                if a["img"] in visited or b["img"] in visited: continue
                if not a["pha"] or not b["pha"]: continue
                d = ham(a["pha"], b["pha"])
                if d <= NEAR_DUP_HAMMING:
                    qa = (a["w"]*a["h"], a["var"])
                    qb = (b["w"]*b["h"], b["var"])
                    keep, drop = (a,b) if qa >= qb else (b,a)
                    move_pair(drop["img"], drop["lbl"], Q/"duplicates_near", f"near_duplicate_h{d}", writer, dry)
                    visited.add(drop["img"]); moved += 1
    print(f"[B] Casi-duplicados movidos: {moved}")
    return moved

def etapa_C_calidad(items, writer, dry):
    m_small=m_blur=m_expo=0
    for it in items:
        if not it["img"].exists(): continue
        if it["w"]<MIN_W or it["h"]<MIN_H:
            move_pair(it["img"], it["lbl"], Q/"too_small", "too_small", writer, dry); m_small+=1; continue
        if it["var"]<MIN_VAR_LAPLACE:
            move_pair(it["img"], it["lbl"], Q/"blurry", "blurry", writer, dry); m_blur+=1; continue
        if it["bri"]<MIN_BRIGHT or it["bri"]>MAX_BRIGHT:
            move_pair(it["img"], it["lbl"], Q/"exposure_review", "exposure_extreme", writer, dry); m_expo+=1; continue
    print(f"[C] Pequeñas: {m_small} | Borrosas: {m_blur} | Exposición extrema: {m_expo}")
    return m_small+m_blur+m_expo

def etapa_D_labels(writer, dry):
    moved=Counter()
    for split in SPLITS:
        img_dir = ROOT/split/"images"
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in IMG_EXTS or not img.exists(): continue
            lbl = img.parent.parent/"labels"/(img.stem+".txt")
            lines = yolo_read(lbl)
            ok, why = yolo_valid(lines)
            if ok: continue
            reason = "bad_label" if why!="tiny_box_only" else "tiny_box"
            dst = Q/("bad_label" if reason=="bad_label" else "tiny_box")
            move_pair(img, lbl, dst, reason, writer, dry)
            moved[reason]+=1
    print(f"[D] Labels movidos -> {dict(moved)}")
    return sum(moved.values())

def count_now():
    def cnt(split):
        return sum(1 for p in (ROOT/split/"images").glob("*") if p.suffix.lower() in IMG_EXTS)
    return {s: cnt(s) for s in SPLITS}

# ====== main ======
def main():
    ap=argparse.ArgumentParser(description="Limpieza por etapas (A:SHA1, B:pHash entre splits, C:calidad, D:labels)")
    ap.add_argument("--only", choices=list("ABCD"), help="Corre solo una etapa")
    ap.add_argument("--from", dest="from_stage", choices=list("ABCD"), help="Corre desde esta etapa en adelante")
    ap.add_argument("--dry-run", action="store_true", help="No mueve archivos, solo registra en log")
    args=ap.parse_args()

    moves_log = LOG_DIR/"moves_log.csv"
    with open(moves_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["reason","src_img","src_lbl","dst_dir","action"])

        stages = [("A", etapa_A_duplicados_exactos),
                  ("B", etapa_B_casi_duplicados),
                  ("C", etapa_C_calidad),
                  ("D", lambda items, w, d: etapa_D_labels(w, d))]

        items = scan()

        run = []
        if args.only:
            run = [s for s in stages if s[0]==args.only]
        elif args.from_stage:
            idx = [i for i,(k,_) in enumerate(stages) if k==args.from_stage][0]
            run = stages[idx:]
        else:
            run = stages

        print("Conteo inicial:", count_now())
        for k, fn in run:
            if k=="D":
                fn(items, writer, args.dry_run)
            else:
                fn(items, writer, args.dry_run)
            print(f"Conteo tras {k}:", count_now(), "\n")

    print("Log:", moves_log)
    print("Cuarentena:", Q)

if __name__=="__main__":
    main()
