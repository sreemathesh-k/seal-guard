# Seal-Guard

Offline blister-pack inspection for small pharma makers (Bharat AI-SoC Challenge 2026-27, PS5).
It learns a product from about 20 good strips only, checks every pocket, turns the findings into a
short operator action (Llama 3.2 1B via ExecuTorch, with a fact check and a fixed-template fallback),
and keeps a tamper-evident log.

## Flow

camera/photo -> find strip -> cut pockets -> colour rules (empty, wrong colour) -> anomaly model
(features + PaDiM, learned from good pockets) -> fact table (JSON) -> advice (Llama or template,
fact-checked) -> hash-chained log -> operator screen. Llama sees only the fact table, never pixels.

## Files

| File | Purpose |
| --- | --- |
| `sealguard/pockets.py` | find and straighten the strip, cut pocket crops |
| `sealguard/rules.py` | empty-pocket and wrong-colour rules |
| `sealguard/backbone.py` | features: classic (OpenCV), torch (ResNet18), pte (ExecuTorch) |
| `sealguard/padim.py` | anomaly model learned from good pockets (NumPy) |
| `sealguard/pipeline.py` | enrolment with automatic threshold; inspection to fact table + overlay |
| `sealguard/advice.py` | prompt, fact check, template fallback, ExecuTorch Llama caller |
| `sealguard/hashlog.py` | SHA-256 hash-chained CSV log and tamper check |
| `enroll.py`, `inspect_strip.py` | learn a product; inspect photos or the Pi camera |
| `advise_strip.py` | advice + log for one fact table; `--verify` checks the log |
| `operator_screen.py` | inspect, advise, log, and write one HTML operator page |
| `run_demo.sh` | the whole flow in one command (synthetic strips) |
| `tests/` | advice/log checks and the synthetic vision self-test |

## Quick start (Linux / WSL2)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    bash run_demo.sh

Real strips (dark matte background, whole strip in view, fixed light):

    python enroll.py --good data/real/good --rows 2 --cols 5 --out models/real --backend classic
    python operator_screen.py --model models/real --images data/real/test

## Status

Results, targets and limits are in the technical report. Synthetic-strip numbers only show the
code works; they are not accuracy claims. Not yet done: Llama on the device, Raspberry Pi timing,
real-strip accuracy.

## Licences

Code dependencies: NumPy, scikit-learn (BSD-3), OpenCV (Apache 2.0), optional PyTorch/ExecuTorch
(BSD). MVTec AD is non-commercial (CC BY-NC-SA 4.0) and is not included. Llama 3.2 is under the
Llama 3.2 Community License; weights are not included.
