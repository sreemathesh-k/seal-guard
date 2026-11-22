"""Advice + tamper-evident log for one inspected strip.

    python3 advise_strip.py --facts runs/inspect/strip_facts.json --log logs/sealguard.csv --llm template
    python3 advise_strip.py --facts ... --log ... --llm executorch --runner ./llama_main --model llama3_2.pte --tokenizer tokenizer.model
    python3 advise_strip.py --verify logs/sealguard.csv
"""
import argparse
import json
import sys
import time

from sealguard import hashlog
from sealguard.advice import ExecuTorchLlama, MockLLM, advise

ap = argparse.ArgumentParser()
ap.add_argument("--facts")
ap.add_argument("--log", default="logs/sealguard.csv")
ap.add_argument("--llm", default="template", choices=["template", "mock", "executorch"])
ap.add_argument("--runner")
ap.add_argument("--model")
ap.add_argument("--tokenizer")
ap.add_argument("--verify")
args = ap.parse_args()

if args.verify:
    ok, msg = hashlog.verify(args.verify)
    print(msg)
    sys.exit(0 if ok else 1)

if not args.facts:
    sys.exit("give --facts <fact table json> or --verify <log>")
facts = json.load(open(args.facts, encoding="utf-8"))
llm = None
if args.llm == "mock":
    llm = MockLLM("good")
elif args.llm == "executorch":
    if not (args.runner and args.model and args.tokenizer):
        sys.exit("--llm executorch needs --runner, --model and --tokenizer")
    llm = ExecuTorchLlama(args.runner, args.model, args.tokenizer)

t0 = time.perf_counter()
res = advise(facts, llm)
ms = 1000 * (time.perf_counter() - t0)
h = hashlog.append(args.log, facts, res["text"], res["source"])
print(f"verdict : {facts.get('verdict')}")
print(f"advice  : {res['text']}")
print(f"source  : {res['source']}   check: {res['check']}   time: {ms:.0f} ms")
print(f"logged  : {args.log}  hash {h[:12]}...")
