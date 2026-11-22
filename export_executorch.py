"""Export the vision models to ExecuTorch .pte files for the Raspberry Pi 5 CPU (XNNPACK).

    # backbone, FP32 and INT8 (INT8 needs real pocket crops for calibration)
    python export_executorch.py backbone --good data/productA/good --rows 2 --cols 5 --out models/productA

    # classifier trained with train_classifier.py
    python export_executorch.py classifier --ckpt models/productA/classifier.pt \
        --good data/productA/good --rows 2 --cols 5 --out models/productA

Run on the laptop (Linux, macOS or WSL2). Copy the .pte files to the Pi.
Written against the ExecuTorch 1.x Python API; if an import fails, check the
XNNPACK quantization page of the docs for your installed version.
"""
import argparse
import json
from pathlib import Path

import cv2
import torch

from sealguard.backbone import build_resnet18_features, to_nchw
from sealguard.pockets import strip_to_pockets


def load_calibration(good_dir, rows, cols, limit=200):
    crops = []
    for p in sorted(Path(good_dir).glob("*")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        crops += strip_to_pockets(img, rows, cols)[1]
        if len(crops) >= limit:
            break
    if not crops:
        raise SystemExit(f"no strips found in {good_dir}")
    return torch.from_numpy(to_nchw(crops[:limit]))


def lower(model, sample, out_path):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    prog = to_edge_transform_and_lower(
        torch.export.export(model, sample), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    Path(out_path).write_bytes(prog.buffer)
    print(f"wrote {out_path}  ({Path(out_path).stat().st_size / 1e6:.2f} MB)")


def quantize_int8(model, sample, calib):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError:  # older ExecuTorch releases
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(model, sample).module(), quantizer)
    with torch.no_grad():
        for i in range(len(calib)):      # calibration: run real pocket crops through
            prepared(calib[i:i + 1])
    return convert_pt2e(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["backbone", "classifier"])
    ap.add_argument("--good", required=True, help="folder of good-strip photos (for INT8 calibration)")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", help="classifier.pt (classifier only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = load_calibration(args.good, args.rows, args.cols)
    sample = (calib[:1],)

    if args.what == "backbone":
        model, name = build_resnet18_features(pretrained=True), "backbone"
    else:
        from sealguard.classifier import build_classifier
        ck = torch.load(args.ckpt, map_location="cpu")
        model = build_classifier(len(ck["classes"]), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        (out / "classes.json").write_text(json.dumps(ck["classes"]))
        name = "classifier"
    model.eval()

    lower(model, sample, out / f"{name}_fp32.pte")
    lower(quantize_int8(model, sample, calib), sample, out / f"{name}_int8.pte")
    print("Benchmark both on the Pi, then copy the one you choose to "
          f"{out / (name + '.pte')} and enrol with --backend pte.")


if __name__ == "__main__":
    main()
