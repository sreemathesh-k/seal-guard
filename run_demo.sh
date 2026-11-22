#!/usr/bin/env bash
# One-command demo of the whole Seal-Guard flow on drawn (synthetic) strips.
# Real strips: replace data/synth/... with your photo folders and --rows/--cols.
set -e
cd "$(dirname "$0")"
echo "== 1. advice fact check and log tests"; python tests/test_advice.py
echo "== 2. vision self-test (synthetic strips)"; python tests/synth_demo.py
if [ ! -d data/synth/good ]; then
  python -c "import cv2,numpy as np,os;from sealguard import synth;r=np.random.default_rng(0);os.makedirs('data/synth/good',exist_ok=True);os.makedirs('data/synth/test',exist_ok=True);[cv2.imwrite(f'data/synth/good/g{i:02d}.png',synth.make_strip(r)) for i in range(20)];cv2.imwrite('data/synth/test/good_strip.png',synth.make_strip(r));cv2.imwrite('data/synth/test/defect_strip.png',synth.make_strip(r,{1:'crack',4:'chip',6:'foreign',8:'wrong_colour'}))"
fi
echo "== 3. learn the product from 20 good strips"; python enroll.py --good data/synth/good --rows 2 --cols 5 --out models/synth_classic --backend classic
echo "== 4. inspect, advise, log, operator screen"; python operator_screen.py --model models/synth_classic --images data/synth/test --out runs/screen.html
echo "== 5. log tamper check"; python advise_strip.py --verify logs/sealguard.csv
