"""Real-photo pilot: find our 15 pilot photos in a folder of phone photos (any names),
copy them as A01..A10 / B01..B05 into train/test folders and write labels.csv.

Photos are matched by an 8x8 grey thumbnail, so file names do not matter.
Strip A = 10-pocket PARACIP strip, plain foil; strip B = same product, printed foil.
Every strip has 4 empty pockets (tablets removed): those are the labelled defects.

    python pilot_sort.py --src /mnt/c/Users/acer/Desktop/SealGuard_photos/all --out data/real
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

FP = {
    "A01": "2d333d454d54595f29386f86939464662a3b778f9c9a65682a3b7b939e9c65652b3a77979b9565672d386a84949865662d375861726f61632b313c474d585d5d",
    "A02": "1536496472939143143f5c667f9e8844123e556c79958b450d405676819892460d43607d98ac8f470f38446d7c8e96491133526f869f9745142f4a4f5a665440",
    "A03": "0c5253686074864607354c5959888246072b4a575b818e460729475e678a9546072a516873a29c4507243b5a6c7a9e47082540576f8bab4a0a254b515b7c6a42",
    "A04": "19223f4a5061544e11295b76779d71520f22506c749a72530e1f51737da17f54101d5471879a8154111a48617a978b54131b475d63826550181c283137424b4e",
    "A05": "111c252a2f363b40101c2730383f4546113d678ea0814847143f6d859d794545113b6a859d7d454412355f849276434411275a6d8c6c3c3d111d33363c3a363a",
    "A06": "514c4233281d1817525f6c665e5b3419556880727364341b555f807c7d6a341e4f528e90877d37284d4d81988e8b463a49496779726e3e3546423f3c3b383532",
    "A07": "3f55665e6657312c3e779e795f5421193e7574695f4f2310428489766a52341147a0887f78523f1453a3897172574011465c504b3f32250f37342f2b2520170e",
    "A08": "0f0f111417202b3719263532435b353a223b54475d70393a243e604a6572373a2554714e6376333927555c44586831391e435641536d32391c212b2a3e3f2f39",
    "A09": "373b3a3a3a3937343759635f63686a43365f777b75696e62326a746f7c7c7c5a2e677b7a7e796a64295f716873916e4c23282d2b2b322e291d1d20201e212521",
    "A10": "4b4d4d4c4d4d4845484c4c4c4d4d49425aa09b9ba19e954a57999593a7889044466f6e737c696a39366765656a4f5f37294d48494e3d3c2b131b181310111215",
    "B01": "15131111111111131533353e454c34211463596f6b6e5d4c1474686e79865c50136a73797a845d4719777c7a87707949223b4447494a4a402a3c4648494a4746",
    "B02": "353b3c3b3e3e3d393c66555262725e41477a756f77866d4f54816a6977726c505c897d7389896461406a545c5d7d794330393939383b3b373032343435343433",
    "B03": "242d5c6a7b7c584a1d2e759b94aa724f182b69897c996b5113265f84809e725211244f6d838f7d531122466857786b51121e38514c5d544e141b263037424a4d",
    "B04": "040b0f11161b1d1f0a3243635a6e5825094c5297918877260837598281916c26073a4c94867f77260624445b6c858d2605172f48556e541c060b171d26331d1a",
    "B05": "07143648464a3d3007325a889295833107264e7b7b806d3007204570737e6d30061f4a8682817431071e3f658191863007183656465d542a070d192325342c28",}
TRAIN = ["A01", "A02", "A03", "A04", "A05", "B01", "B02", "B03"]
FLIPPED = {"A06", "A07"}  # photographed upside down: empty pockets come out on the left

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="data/real")
args = ap.parse_args()

ref = {k: np.frombuffer(bytes.fromhex(v), np.uint8).astype(float) for k, v in FP.items()}
found = {}
for p in sorted(Path(args.src).iterdir()):
    img = cv2.imread(str(p))
    if img is None:
        continue
    f = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA).astype(float).ravel()
    name, d = min(((k, np.abs(f - v).mean()) for k, v in ref.items()), key=lambda t: t[1])
    if d < 6 and (name not in found or d < found[name][1]):
        found[name] = (p, d)

out = Path(args.out)
for sub in ("train", "test"):
    (out / sub).mkdir(parents=True, exist_ok=True)
rows = ["image,pocket,defect"]
for name in sorted(found):
    p = found[name][0]
    dst = f"{name}{p.suffix.lower()}"
    shutil.copy(p, out / ("train" if name in TRAIN else "test") / dst)
    rows += [f"{dst},{q},empty" for q in ([1, 2, 6, 7] if name in FLIPPED else [4, 5, 9, 10])]
(out / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"matched {len(found)}/15 pilot photos: {' '.join(sorted(found))}")
missing = sorted(set(FP) - set(found))
if missing:
    print("NOT found:", " ".join(missing))
