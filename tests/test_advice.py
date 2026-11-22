"""10 checks for the advice fact check and the hash log. Plain Python.

    python3 tests/test_advice.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealguard import hashlog
from sealguard.advice import MockLLM, advise, fact_check, template_advice

FAIL = {"verdict": "FAIL", "pockets": [
    {"pocket": 1, "verdict": "PASS", "type": None, "severity": None, "area_pct": 0.0},
    {"pocket": 4, "verdict": "FAIL", "type": "anomaly", "severity": "critical", "area_pct": 6.2},
    {"pocket": 8, "verdict": "FAIL", "type": "empty", "severity": "critical", "area_pct": 40.0}]}
PASS = {"verdict": "PASS", "pockets": [{"pocket": 1, "verdict": "PASS"}]}


class Boom:
    def __call__(self, prompt, facts):
        raise RuntimeError("runner not found")


checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


r = advise(PASS, Boom())
check("1 PASS strip never calls the LLM", r["source"] == "none")
t = template_advice(FAIL)
check("2 template names every failed pocket and passes the check", fact_check(t, FAIL)[0])
check("3 faithful LLM reply is accepted", advise(FAIL, MockLLM("good"))["source"] == "mock")
r = advise(FAIL, MockLLM("invent"))
check("4 invented pocket is rejected, template used", r["source"] == "template" and "99" in r["check"])
check("5 'strip is fine' claim is rejected", advise(FAIL, MockLLM("fine"))["source"] == "template")
check("6 leaving out a failed pocket is rejected", not fact_check("Reject: pocket 4 anomaly.", FAIL)[0])
check("7 defect type not in facts is rejected", not fact_check("Reject: pocket 4 has a crack, pocket 8 empty.", FAIL)[0])
check("8 invented percentage is rejected", not fact_check("Reject: pocket 4 anomaly 12% area, pocket 8 empty.", FAIL)[0])
r = advise(FAIL, Boom())
check("9 crashed LLM falls back to template", r["source"] == "template" and "runner not found" in r["check"])

d = tempfile.mkdtemp()
log = os.path.join(d, "log.csv")
for _ in range(3):
    hashlog.append(log, FAIL, template_advice(FAIL), "template")
ok_before = hashlog.verify(log)[0]
txt = Path(log).read_text(encoding="utf-8").replace("Reject this strip", "Accept this strip", 1)
Path(log).write_text(txt, encoding="utf-8")
ok_after, msg = hashlog.verify(log)
check("10 hash log verifies, and an edited row is detected", ok_before and not ok_after and "row 1" in msg)

n = sum(ok for _, ok in checks)
print(f"\n{n}/{len(checks)} checks passed")
sys.exit(0 if n == len(checks) else 1)
