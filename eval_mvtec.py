"""Image-level AUROC on MVTec AD (deck target: >= 0.95 on pill and capsule).

Download MVTec AD yourself (licence: CC BY-NC-SA 4.0) and point --root at it:
    <root>/pill/train/good/*.png
    <root>/pill/test/<defect or good>/*.png

    python eval_mvtec.py --root data/mvtec                                   # PaDiM + ResNet18 (fast)
    python eval_mvtec.py --root data/mvtec --method patchcore --arch resnet18
    python eval_mvtec.py --root data/mvtec --method patchcore --arch wide_resnet50_2   # most accurate
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from sealguard.backbone import TorchBackbone, build_patchcore_features, make_backbone
from sealguard.padim import Padim
from sealguard.patchcore import PatchCore

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--categories", nargs="+", default=["pill", "capsule"])
ap.add_argument("--method", default="padim", choices=["padim", "patchcore"])
ap.add_argument("--arch", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
ap.add_argument("--resize", type=int, default=256)
ap.add_argument("--crop", type=int, default=224)
ap.add_argument("--coreset", type=float, default=0.01, help="PatchCore memory bank size (fraction)")
args = ap.parse_args()

if args.method == "padim":
    bb = make_backbone("torch")
    new_model = Padim
else:
    out_dim = 512 if args.arch == "wide_resnet50_2" else None
    bb = TorchBackbone(model=build_patchcore_features(args.arch, out_dim=out_dim), batch=16)
    def new_model():
        return PatchCore(ratio=args.coreset)


def load(folder):
    """Resize to --resize, then centre-crop to --crop (the usual MVTec protocol)."""
    paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".png", ".jpg"))
    out = []
    o = (args.resize - args.crop) // 2
    for p in paths:
        img = cv2.resize(cv2.imread(str(p)), (args.resize, args.resize), interpolation=cv2.INTER_AREA)
        out.append(img[o:o + args.crop, o:o + args.crop])
    return out


print(f"method {args.method}, backbone {args.arch if args.method == 'patchcore' else 'resnet18'}")
for cat in args.categories:
    root = Path(args.root) / cat
    train = load(root / "train" / "good")
    if not train:
        sys.exit(f"no training images in {root / 'train' / 'good'} - check --root and that the archive is unpacked")
    t0 = time.time()
    model = new_model().fit(bb(train))
    t_fit = time.time() - t0
    y, s, t_inf = [], [], 0.0
    for d in sorted((root / "test").iterdir()):
        imgs = load(d)
        if not imgs:
            continue
        t1 = time.time()
        s += list(model.scores(model.score_maps(bb(imgs))))
        t_inf += time.time() - t1
        y += [d.name != "good"] * len(imgs)
    print(f"{cat:<10} image AUROC {roc_auc_score(y, s):.4f}   ({sum(y)} defective, {len(y) - sum(y)} good)"
          f"   fit {t_fit:.0f} s, {1000 * t_inf / len(y):.0f} ms/image on this laptop")
