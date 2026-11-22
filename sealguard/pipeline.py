"""Enrolment (learn a product from good strips) and inspection (check a strip)."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import rules as R
from .backbone import make_backbone
from .padim import Padim
from .pockets import crop_pockets, find_strip

CRITICAL_TYPES = {"empty", "wrong_colour"}


def severity(defect_type, area_pct):
    """Written severity rule (edit to match your QA policy).

    critical : missing or wrong tablet, or damage covering >= 5 % of the pocket
    major    : any crack, or damage covering 1-5 %
    minor    : everything smaller
    """
    if defect_type in CRITICAL_TYPES or area_pct >= 5.0:
        return "critical"
    if defect_type == "crack" or area_pct >= 1.0:
        return "major"
    return "minor"


def enroll(images, rows, cols, out_dir, backend="classic", pte_path=None, folds=5, k=6.0, seed=0):
    """Learn one product from images of GOOD strips. Returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops, strip_id = [], []
    for n, img in enumerate(images):
        strip = find_strip(img)
        if strip is None:
            continue
        c = crop_pockets(strip, rows, cols)[0]
        crops += c
        strip_id += [n] * len(c)
    if len(crops) < 20:
        raise ValueError(f"only {len(crops)} good pockets found; need at least 20")
    bb = make_backbone(backend, pte_path)
    feats = bb(crops)
    # Threshold from cross-validation over whole strips: every good pocket gets a
    # score from a model that never saw its strip (pockets of one strip share
    # lighting and alignment, so a per-pocket split would be over-optimistic).
    strip_id = np.array(strip_id)
    ids = np.random.default_rng(seed).permutation(np.unique(strip_id))
    cv_scores = np.zeros(len(crops), np.float32)
    for fold in np.array_split(ids, min(folds, len(ids))):
        hold = np.isin(strip_id, fold)
        m = Padim().fit(feats[~hold])
        cv_scores[hold] = m.scores(m.score_maps(feats[hold]))
    model = Padim().fit(feats)
    thr = model.set_threshold(cv_scores, k=k)
    model.save(out_dir / "padim.npz")
    meta = {"rows": rows, "cols": cols, "backend": backend, "threshold": thr,
            "good_pockets": len(crops), "rules": R.fit_rules(crops)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class Inspector:
    def __init__(self, model_dir, backend=None, pte_path=None, classifier=None):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.model = Padim.load(model_dir / "padim.npz")
        if pte_path is None and (model_dir / "backbone.pte").exists():
            pte_path = model_dir / "backbone.pte"
        self.bb = make_backbone(backend or self.meta["backend"], pte_path)
        self.classifier = classifier  # optional callable: crop -> (type, confidence)

    def inspect(self, img):
        """Return the fact table for one strip (this is what Llama is allowed to see)."""
        t0 = time.perf_counter()
        rows, cols = self.meta["rows"], self.meta["cols"]
        strip = find_strip(img)
        if strip is None:
            return {"verdict": "NO_STRIP", "pockets": [], "ms": 0.0}, None
        crops, boxes = crop_pockets(strip, rows, cols)
        maps = self.model.score_maps(self.bb(crops))
        scores = self.model.scores(maps)
        thr = self.model.thr
        pockets = []
        for i, crop in enumerate(crops):
            rule = R.apply_rules(crop, self.meta["rules"])
            area = 100.0 * float((maps[i] > thr).mean())
            failed = rule is not None or scores[i] > thr
            dtype, conf = None, None
            if failed:
                if rule:
                    dtype, conf = rule, 1.0
                elif self.classifier:
                    dtype, conf = self.classifier(crop)
                else:
                    dtype = "anomaly"
            pockets.append({
                "pocket": i + 1, "verdict": "FAIL" if failed else "PASS",
                "type": dtype, "confidence": conf,
                "severity": severity(dtype, area) if failed else None,
                "area_pct": round(area, 2), "score": round(float(scores[i]), 3),
            })
        facts = {
            "verdict": "FAIL" if any(p["verdict"] == "FAIL" for p in pockets) else "PASS",
            "threshold": round(thr, 3), "pockets": pockets,
            "ms": round(1000 * (time.perf_counter() - t0), 1),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return facts, self._overlay(strip, boxes, maps, pockets, thr)

    @staticmethod
    def _overlay(strip, boxes, maps, pockets, thr):
        out = strip.copy()
        for (x0, y0, x1, y1), m, p in zip(boxes, maps, pockets):
            if p["verdict"] == "FAIL":
                heat = cv2.resize(np.clip(m / (2 * thr), 0, 1), (x1 - x0, y1 - y0))
                colour = cv2.applyColorMap((255 * heat).astype(np.uint8), cv2.COLORMAP_JET)
                out[y0:y1, x0:x1] = cv2.addWeighted(out[y0:y1, x0:x1], 0.55, colour, 0.45, 0)
            col = (0, 0, 255) if p["verdict"] == "FAIL" else (0, 170, 0)
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
            label = f'{p["pocket"]}: {p["type"]}' if p["type"] else str(p["pocket"])
            cv2.putText(out, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return out
