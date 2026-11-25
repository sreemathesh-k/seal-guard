import json, time
from sealguard.advice import ExecuTorchLlama, build_prompt, fact_check
import os; H = os.path.expanduser("~") + "/"
L = ExecuTorchLlama(H + "executorch-rpi/executorch/cmake-out/examples/models/llama/llama_main",
                    H + "llama1b/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte", H + "llama1b/tokenizer.model")
def F(*bad):
    ps = [{"pocket": n, "verdict": "PASS", "type": None, "severity": None, "area_pct": 0.0} for n in range(1, 11)]
    for n, t, sv in bad:
        ps[n - 1].update(verdict="FAIL", type=t, severity=sv, area_pct=5.0)
    return {"verdict": "FAIL", "pockets": ps}
cases = [("e2e defect strip", json.load(open("runs/e2e/defect_strip_facts.json"))),
         ("two empty", F((4, "empty", "critical"), (8, "empty", "critical"))),
         ("one minor anomaly", F((1, "anomaly", "minor"),)),
         ("colour + anomaly", F((3, "wrong_colour", "critical"), (6, "anomaly", "major"))),
         ("last pocket empty", F((10, "empty", "critical"),)),
         ("three mixed", F((2, "anomaly", "major"), (5, "anomaly", "minor"), (9, "empty", "critical")))]
ok_n = 0
for name, facts in cases:
    t0 = time.time(); r = L(build_prompt(facts), facts); ms = (time.time() - t0) * 1000
    ok, why = fact_check(r, facts); ok_n += ok; s = L.stats or {}
    ft = s.get("first_token_ms", 0) - s.get("inference_start_ms", 0)
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} ({why}) call {ms:.0f} ms, first token {ft} ms, "
          f"{s.get('generated_tokens')} tokens, decode {s.get('decode_token_per_sec', 0):.1f} tok/s")
    print(r.replace("\n", " | "))
print(f"PASS RATE: {ok_n}/{len(cases)} fact tables (temperature 0, x86 laptop CPU, not Arm)")
