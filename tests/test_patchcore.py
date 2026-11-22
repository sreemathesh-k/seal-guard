"""PatchCore check on synthetic pockets with OpenCV features (no PyTorch needed).

    python tests/test_patchcore.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealguard import synth
from sealguard.backbone import ClassicBackbone
from sealguard.patchcore import PatchCore
from sealguard.pockets import strip_to_pockets

rng, bb = np.random.default_rng(3), ClassicBackbone()
good = [c for _ in range(20) for c in strip_to_pockets(synth.make_strip(rng), 2, 5)[1]]
pc = PatchCore(min_bank=500).fit(bb(good))
y, s = [], []
for i in range(60):
    defects = {} if i < 30 else {int(rng.integers(10)): ["crack", "chip", "foreign"][i % 3]}
    crops = strip_to_pockets(synth.make_strip(rng, defects), 2, 5)[1]
    s += list(pc.scores(pc.score_maps(bb(crops))))
    y += [k in defects for k in range(10)]
print(f"PatchCore bank {pc.bank.shape}, synthetic pocket AUROC {roc_auc_score(y, s):.4f}")
