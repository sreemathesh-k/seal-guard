"""Score inspection results against hand-written labels: misses vs false alarms.

labels.csv (one row per DEFECTIVE pocket; strips not listed are all good):
    image,pocket,defect
    strip07.jpg,4,crack
    strip07.jpg,8,empty

    python score_real.py --model models/real --images data/real/test --labels data/real/labels.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from sealguard.pipeline import Inspector

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="runs/scores.txt")
args = ap.parse_args()

truth = defaultdict(dict)
with open(args.labels, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        truth[r["image"].strip()][int(r["pocket"])] = r["defect"].strip()

insp = Inspector(args.model)
tp = fp = fn = tn = 0
hit, total, strips_ok, n_strips, ms = Counter(), Counter(), 0, 0, []
lines = []
for p in sorted(Path(args.images).glob("*")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    facts, _ = insp.inspect(img)
    if facts["verdict"] == "NO_STRIP":
        lines.append(f"{p.name}: strip not found")
        continue
    n_strips += 1
    ms.append(facts["ms"])
    lab = truth.get(p.name, {})
    for q in facts["pockets"]:
        bad, said = q["pocket"] in lab, q["verdict"] == "FAIL"
        tp += bad and said
        fn += bad and not said
        fp += said and not bad
        tn += not bad and not said
        if bad:
            total[lab[q["pocket"]]] += 1
            hit[lab[q["pocket"]]] += said
    strip_bad = bool(lab)
    strips_ok += strip_bad == (facts["verdict"] == "FAIL")
    lines.append(f"{p.name}: {facts['verdict']}  labelled {sorted(lab.items()) or 'good'}  "
                 f"flagged {[q['pocket'] for q in facts['pockets'] if q['verdict'] == 'FAIL']}")

prec = tp / (tp + fp) if tp + fp else float("nan")
rec = tp / (tp + fn) if tp + fn else float("nan")
f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
out = lines + ["",
    f"strips scored        : {n_strips}   strip verdict correct: {strips_ok}/{n_strips}",
    f"pockets              : TP {tp}  FP {fp}  FN {fn}  TN {tn}",
    f"precision / recall   : {prec:.3f} / {rec:.3f}   F1 {f1:.3f}",
    f"false-reject rate    : {fp}/{fp + tn} good pockets ({100 * fp / max(fp + tn, 1):.2f} %)"]
out += [f"recall {k:<14}: {hit[k]}/{total[k]}" for k in sorted(total)]
if ms:
    out.append(f"time per strip       : p50 {np.percentile(ms, 50):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms (this machine)")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"saved: {args.out}")
