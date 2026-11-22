"""Inspect strip photos (or the live Pi camera) with an enrolled product model.

    python inspect_strip.py --model models/productA --images data/productA/test
    python inspect_strip.py --model models/productA --camera            # on the Pi

Prints the fact table per strip and saves <name>_overlay.png next to --save.
With --gpio (Pi only) a FAIL pulses the reject pin / buzzer.
"""
import argparse
import json
from pathlib import Path

import cv2

from sealguard.pipeline import Inspector

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--images")
ap.add_argument("--camera", action="store_true")
ap.add_argument("--backend", choices=["classic", "torch", "pte"])
ap.add_argument("--classifier", help="classifier.pt or classifier.pte")
ap.add_argument("--save", default="runs/inspect")
ap.add_argument("--gpio", type=int, help="BCM pin for buzzer / reject signal")
args = ap.parse_args()

clf = None
if args.classifier:
    if args.classifier.endswith(".pte"):
        from sealguard.classifier import PteClassifier
        classes = json.loads((Path(args.classifier).parent / "classes.json").read_text())
        clf = PteClassifier(args.classifier, classes)
    else:
        from sealguard.classifier import TorchClassifier
        clf = TorchClassifier(args.classifier)

insp = Inspector(args.model, backend=args.backend, classifier=clf)
save = Path(args.save)
save.mkdir(parents=True, exist_ok=True)
reject = None
if args.gpio is not None:
    from gpiozero import DigitalOutputDevice   # works on Pi 5 (uses lgpio)
    reject = DigitalOutputDevice(args.gpio)


def handle(name, frame):
    facts, overlay = insp.inspect(frame)
    bad = [f'{p["pocket"]}:{p["type"]}({p["severity"]})' for p in facts["pockets"] if p["verdict"] == "FAIL"]
    print(f'{name}: {facts["verdict"]}  {facts["ms"]} ms  {" ".join(bad)}')
    if overlay is not None:
        cv2.imwrite(str(save / f"{name}_overlay.png"), overlay)
        (save / f"{name}_facts.json").write_text(json.dumps(facts, indent=2))
    if reject and facts["verdict"] == "FAIL":
        reject.blink(on_time=0.3, off_time=0.1, n=1)
    return facts


if args.camera:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1440, 960), "format": "RGB888"}))
    # fixed exposure and white balance: auto modes would look like defects to the model
    cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": 8000,
                      "AnalogueGain": 1.0, "ColourGains": (1.6, 1.6)})
    cam.start()
    n = 0
    print("press Enter to inspect a strip, Ctrl+C to stop")
    try:
        while True:
            input()
            n += 1
            handle(f"strip_{n:05d}", cam.capture_array())  # picamera2 'RGB888' is BGR order, as OpenCV expects
    except KeyboardInterrupt:
        cam.stop()
else:
    for p in sorted(Path(args.images).glob("*")):
        img = cv2.imread(str(p))
        if img is not None:
            handle(p.stem, img)
