"""End-to-end self-test on synthetic strips. Needs only numpy, opencv, scikit-learn.

    python tests/synth_demo.py [--backend classic|torch]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealguard import synth
from sealguard.pipeline import Inspector, enroll

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="classic")
ap.add_argument("--out", default="runs/synth")
args = ap.parse_args()
out = Path(args.out)
rng = np.random.default_rng(1)

meta = enroll([synth.make_strip(rng) for _ in range(20)], synth.ROWS, synth.COLS, out, backend=args.backend)
print(f"enrolled {meta['good_pockets']} good pockets, threshold {meta['threshold']:.2f}")

insp = Inspector(out)
y_true, y_score = [], []
hit, total, types = Counter(), Counter(), defaultdict(Counter)
false_rej = good = 0
times = []
for s in range(150):
    defects = {}
    if s >= 50:  # 50 good strips, 100 strips with one or two defects
        for p in rng.choice(10, int(rng.integers(1, 3)), replace=False):
            defects[int(p)] = synth.DEFECTS[int(rng.integers(len(synth.DEFECTS)))]
    facts, overlay = insp.inspect(synth.make_strip(rng, defects))
    times.append(facts["ms"])
    for p in facts["pockets"]:
        kind = defects.get(p["pocket"] - 1)
        y_true.append(kind is not None)
        y_score.append(1e6 if p["type"] in ("empty", "wrong_colour") else p["score"])
        if kind:
            total[kind] += 1
            hit[kind] += p["verdict"] == "FAIL"
            types[kind][p["type"]] += 1
        else:
            good += 1
            false_rej += p["verdict"] == "FAIL"
    if s == 50:
        cv2.imwrite(str(out / "example_overlay.png"), overlay)
        (out / "example_facts.json").write_text(json.dumps(facts, indent=2))
        print("example defects:", {k + 1: v for k, v in defects.items()})

print(f"\npocket AUROC      : {roc_auc_score(y_true, y_score):.4f}")
print(f"false rejects     : {false_rej}/{good} good pockets ({100 * false_rej / good:.2f} %)")
for k in synth.DEFECTS:
    print(f"recall {k:<12}: {hit[k]}/{total[k]}  reported as {dict(types[k])}")
print(f"time per strip    : p50 {np.percentile(times, 50):.0f} ms, p95 {np.percentile(times, 95):.0f} ms (this machine)")
