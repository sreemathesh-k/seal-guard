import csv, cv2, numpy as np, sys
from sealguard.pockets import strip_to_pockets
def crops(path):
    c = strip_to_pockets(cv2.imread(path), 2, 5)
    return [np.asarray(x) for x in (c[1] if isinstance(c, tuple) else c)]
def feats(cr):
    n = cr.shape[0]; a, b = int(0.3 * n), int(0.7 * n); m = cr[a:b, a:b]
    hsv = cv2.cvtColor(m, cv2.COLOR_BGR2HSV).astype(float)
    g = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return {"V_mean": hsv[..., 2].mean(), "S_mean": hsv[..., 1].mean(),
            "V_std": hsv[..., 2].std(), "edges": (cv2.Canny(g, 50, 150) > 0).mean() * 100,
            "dark_frac": (hsv[..., 2] < 170).mean() * 100}
lab = {}
for r in csv.DictReader(open("data/real/labels.csv", encoding="utf-8-sig")):
    lab.setdefault(r["image"], {})[int(r["pocket"])] = r["defect"]
split = sys.argv[1]
rows = []
for img in sorted(lab):
    import os
    p = f"data/real/{split}/{img}"
    if not os.path.exists(p): continue
    for i, cr in enumerate(crops(p), 1):
        rows.append((img, i, lab[img].get(i, "good"), feats(cr)))
print(split, "photos:", len({r[0] for r in rows}), "pockets:", len(rows),
      "empty:", sum(r[2] == "empty" for r in rows))
for k in rows[0][3]:
    g = np.array([r[3][k] for r in rows if r[2] == "good"])
    e = np.array([r[3][k] for r in rows if r[2] == "empty"])
    print(f"{k:9s} good min {g.min():6.1f} med {np.median(g):6.1f} max {g.max():6.1f} | "
          f"empty min {e.min():6.1f} med {np.median(e):6.1f} max {e.max():6.1f}")
np.save(f"/tmp/feat_{split}.npy", np.array([[r[1], r[2] == "empty"] + list(r[3].values()) for r in rows]))
