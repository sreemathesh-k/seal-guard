"""Learn a new product from photos of GOOD strips only (about 20 strips).

    python enroll.py --good data/productA/good --rows 2 --cols 5 --out models/productA --backend torch

Backends: classic (no PyTorch), torch (laptop, FP32), pte (ExecuTorch; expects
models/productA/backbone.pte or --pte). Enrol with the backend you will inspect with.
"""
import argparse
import time
from pathlib import Path

import cv2

from sealguard.pipeline import enroll

ap = argparse.ArgumentParser()
ap.add_argument("--good", required=True)
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--cols", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="torch", choices=["classic", "torch", "pte"])
ap.add_argument("--pte")
ap.add_argument("--k", type=float, default=6.0, help="threshold strictness; lower = stricter")
args = ap.parse_args()

images = [im for im in (cv2.imread(str(p)) for p in sorted(Path(args.good).glob("*"))) if im is not None]
pte = args.pte or (str(Path(args.out) / "backbone.pte") if args.backend == "pte" else None)
t0 = time.time()
meta = enroll(images, args.rows, args.cols, args.out, backend=args.backend, pte_path=pte, k=args.k)
print(f"enrolled {meta['good_pockets']} good pockets from {len(images)} photos in {time.time() - t0:.1f} s")
print(f"threshold {meta['threshold']:.3f}  ->  {args.out}")
