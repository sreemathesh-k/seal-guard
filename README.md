# Seal-Guard

Offline blister-strip inspection with on-device advice (Bharat AI-SoC Challenge 2026-27, PS5).
It learns a product from photos of good strips only, checks every pocket, turns the findings into
a short action for the operator with Llama 3.2 1B running through ExecuTorch, and keeps a
tamper-evident log of every decision.

## How it works

photo -> find and straighten strip -> cut pockets -> colour rules (empty, wrong colour) ->
anomaly model (features + PaDiM, learned from good pockets) -> fact table (JSON) ->
Llama advice, checked against the facts (template if the check fails) -> hash-chained log ->
operator page. Llama sees only the fact table, never the image. A PASS strip never calls Llama.

## Results (all measured on an x86 laptop CPU, not yet on Arm hardware)

| What | Result | Raw output |
| --- | --- | --- |
| Llama 3.2 1B via ExecuTorch, offline | runs | `results/llama_run.txt` |
| Llama replies passing the fact check | 5 of 6 (the 6th invented pockets and was blocked) | `results/llama_passrate.txt` |
| Llama first token / decode | 2.7-3.6 s / 13.6-14.8 tok/s (target 2 s missed on this laptop) | same |
| Real photos: good pockets wrongly rejected | 0 of 42 | `results/real_classic.txt` |
| Real photos: empty pockets found | 4 of 28 (weak) | same |
| Texture rule for empties (experiment) | 22 of 28 found, 11 of 42 good rejected: not adopted | `results/empty_rule_test.txt` |
| MVTec AD image AUROC, pill / capsule | 0.875 / 0.893 (target 0.95) | earlier run of `eval_mvtec.py` |
| Tamper test (one edited row) | detected | `bash demo_video.sh` |

The real-photo pilot is small (2 strips, 15 hand-held phone photos, 8 for learning, 7 for testing).
The photos are overexposed, which hides empty pockets; a light box is the next step.
Synthetic-strip results only show that the code runs; they are not accuracy claims.

## Files

| File | Purpose |
| --- | --- |
| `sealguard/pockets.py` | find and straighten the strip, cut pocket crops |
| `sealguard/rules.py` | empty-pocket and wrong-colour rules |
| `sealguard/backbone.py` | features: classic (OpenCV), torch (ResNet18), pte (ExecuTorch) |
| `sealguard/padim.py`, `patchcore.py` | anomaly models learned from good pockets (NumPy) |
| `sealguard/pipeline.py` | enrolment with automatic threshold; inspection to fact table + overlay |
| `sealguard/advice.py` | prompt, fact check, template fallback, ExecuTorch Llama caller |
| `sealguard/hashlog.py` | SHA-256 hash-chained CSV log and tamper check |
| `enroll.py`, `inspect_strip.py` | learn a product; inspect photos (camera option untested) |
| `advise_strip.py` | advice + log for one fact table; `--verify` checks the log |
| `operator_screen.py` | inspect, advise, log, and write one HTML operator page |
| `score_real.py`, `pilot_sort.py` | score real photos against `labels.csv`; sort pilot photos |
| `llama_passrate.py` | Llama fact-check pass rate over 6 fact tables |
| `experiments/` | the texture-rule experiment (chosen on training photos, tested once) |
| `eval_mvtec.py`, `export_executorch.py` | MVTec AD evaluation; FP32/INT8 ExecuTorch export |
| `run_demo.sh`, `demo_video.sh` | synthetic end-to-end demo; the 4-part demo used in our video |
| `tests/` | advice/log checks (10) and synthetic vision self-tests |

## Quick start (Linux / WSL2)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    bash run_demo.sh
    python tests/test_advice.py

Llama needs the ExecuTorch `llama_main` runner (built from pytorch/executorch, release/1.5,
`make llama-cpu`) and a Llama 3.2 1B `.pte` plus `tokenizer.model`; see `advise_strip.py --help`.
Model weights and photos are not in this repository.

## Licences

NumPy, scikit-learn (BSD-3), OpenCV (Apache 2.0), PyTorch/ExecuTorch (BSD). MVTec AD is
non-commercial (CC BY-NC-SA 4.0) and is not included. Llama 3.2 is under the Llama 3.2 Community
License; weights are not included. The Llama `.pte` we ran is the pre-exported
Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8 file from huggingface.co/executorch-community.

Built with Llama.

Technical report: `docs/SealGuard_Phase1_Report.pdf`. Demo video: https://youtu.be/KLnRrnbry9Y
