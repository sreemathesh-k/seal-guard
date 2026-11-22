"""Operator screen: inspect strips, get advice, write the log, and show it all as one web page.

    python operator_screen.py --model models/productA --images data/productA/test
    python operator_screen.py ... --llm executorch --runner ./llama_main --model-pte llama3_2.pte --tokenizer tokenizer.model
Open the printed .html file in any browser (on Windows: explorer.exe <file>).
"""
import argparse
import base64
import html
import time
from pathlib import Path

import cv2

from sealguard import hashlog
from sealguard.advice import ExecuTorchLlama, advise
from sealguard.pipeline import Inspector

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--log", default="logs/sealguard.csv")
ap.add_argument("--out", default="runs/screen.html")
ap.add_argument("--llm", default="template", choices=["template", "executorch"])
ap.add_argument("--runner")
ap.add_argument("--model-pte")
ap.add_argument("--tokenizer")
args = ap.parse_args()

llm = ExecuTorchLlama(args.runner, args.model_pte, args.tokenizer) if args.llm == "executorch" else None
insp = Inspector(args.model)
cards = []
for p in sorted(Path(args.images).glob("*")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    facts, overlay = insp.inspect(img)
    t0 = time.perf_counter()
    res = advise(facts, llm)
    adv_ms = 1000 * (time.perf_counter() - t0)
    h = hashlog.append(args.log, facts, res["text"], res["source"])
    ok = facts["verdict"] == "PASS"
    pic = ""
    if overlay is not None:
        pic = base64.b64encode(cv2.imencode(".png", overlay)[1].tobytes()).decode()
    rows = "".join(f"<tr><td>{q['pocket']}</td><td>{q['type']}</td><td>{q['severity']}</td><td>{q['area_pct']}</td></tr>"
                   for q in facts["pockets"] if q["verdict"] == "FAIL")
    cards.append(f"""<div class=card><h2 class={'pass' if ok else 'fail'}>{html.escape(p.stem)}: {facts['verdict']}</h2>
<img src="data:image/png;base64,{pic}">
{'<table><tr><th>Pocket</th><th>Defect</th><th>Severity</th><th>Area %</th></tr>' + rows + '</table>' if rows else ''}
<p class=adv><b>Action:</b> {html.escape(res['text'])}</p>
<p class=meta>advice source: {res['source']} ({html.escape(res['check'])}) &middot; vision {facts['ms']} ms &middot; advice {adv_ms:.0f} ms &middot; log hash {h[:12]}</p></div>""")

ok, msg = hashlog.verify(args.log)
page = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Seal-Guard operator screen</title><style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f4f6f8;color:#1b2a3a}} .card{{background:#fff;border-radius:8px;padding:16px;margin:0 0 16px;max-width:760px}}
h2{{margin:0 0 8px}} .pass{{color:#2e7d32}} .fail{{color:#c62828}} img{{max-width:100%;border-radius:4px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #d5dde0;padding:4px 10px;text-align:left}}
.adv{{font-size:16px}} .meta{{color:#5f6b7a;font-size:12px}} .log{{font-weight:bold;color:{'#2e7d32' if ok else '#c62828'}}}
</style></head><body><h1>Seal-Guard operator screen</h1><p class=log>Inspection log: {html.escape(msg)}</p>{''.join(cards)}</body></html>"""
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(page, encoding="utf-8")
print(f"{len(cards)} strips inspected; log: {msg}")
print(f"screen: {args.out}")
