"""Turn a strip's fact table into operator advice, safely.

Flow: facts -> prompt -> LLM (Llama 3.2 1B via ExecuTorch) -> fact check -> reply,
or a fixed template if the LLM is unavailable or its reply fails the check.
A PASS strip never calls the LLM. Plain Python: no PyTorch needed.
"""
import re
import subprocess

DEFECT_WORDS = {
    "empty": ["empty", "missing tablet", "missing"],
    "wrong_colour": ["wrong colour", "wrong color", "discolour", "discolor"],
    "crack": ["crack"],
    "chip": ["chip"],
    "foreign": ["foreign", "contaminat", "speck", "particle"],
    "anomaly": ["anomaly", "abnormal"],
}
ACTION = {
    "empty": "check the feeder for a skipped tablet",
    "wrong_colour": "quarantine the batch and check for product mix-up",
    "crack": "check tablet hardness and the forming/sealing pressure",
    "chip": "check tablet handling and the feeder for impact damage",
    "foreign": "stop and clean the line, then inspect the area for contamination",
    "anomaly": "inspect the pocket by eye and record what you see",
}
FINE_CLAIMS = ["strip is fine", "no defect", "no issues", "all pockets are good", "strip passed", "acceptable"]
MAX_CHARS = 600


def failed_pockets(facts):
    return [p for p in facts.get("pockets", []) if p.get("verdict") == "FAIL"]


def build_prompt(facts):
    lines = [f"pocket {p['pocket']}: {p.get('type') or 'anomaly'}, severity {p.get('severity') or 'unknown'}"
             for p in failed_pockets(facts)]
    return ("You assist a pharmaceutical packaging operator. A blister strip FAILED inspection.\n"
            "Facts (use only these; do not add pockets, defects or numbers):\n- " + "\n- ".join(lines) +
            "\nIn at most 3 short sentences: say the strip must be rejected, name each failed pocket "
            "and its defect, and give one practical check for the line.")


def template_advice(facts):
    bad = failed_pockets(facts)
    parts = [f"pocket {p['pocket']} {(p.get('type') or 'anomaly').replace('_', ' ')} ({p.get('severity') or 'unknown'})" for p in bad]
    types = []
    for p in bad:
        t = p.get("type") or "anomaly"
        if t not in types:
            types.append(t)
    checks = "; ".join(ACTION.get(t, ACTION["anomaly"]) for t in types)
    return f"Reject this strip: {', '.join(parts)}. Next: {checks}."


def fact_check(text, facts):
    """Return (ok, reason). Blocks invented pockets/defects/percentages, 'fine' claims, omissions, long replies."""
    low = text.lower()
    bad = failed_pockets(facts)
    allowed = {int(p["pocket"]) for p in bad}
    if len(text) > MAX_CHARS:
        return False, f"too long ({len(text)} chars)"
    if any(c in low for c in FINE_CLAIMS):
        return False, "claims the strip is fine"
    mentioned = {int(n) for n in re.findall(r"pockets?\s*(?:no\.?|number|#)?\s*(\d+)", low)}
    mentioned |= {int(n) for grp in re.findall(r"pockets?\s+((?:\d+\s*(?:,|and|&)\s*)+\d+)", low) for n in re.findall(r"\d+", grp)}
    if mentioned - allowed:
        return False, f"mentions pockets not in the facts: {sorted(mentioned - allowed)}"
    if allowed - mentioned:
        return False, f"leaves out failed pockets: {sorted(allowed - mentioned)}"
    types = {p.get("type") or "anomaly" for p in bad}
    for t, words in DEFECT_WORDS.items():
        if t in types or t == "anomaly":
            continue
        if any(w in low for w in words):
            return False, f"mentions a defect type not in the facts: {t}"
    facts_pct = {round(float(p.get("area_pct", -1)), 1) for p in bad}
    for n in re.findall(r"(\d+(?:\.\d+)?)\s*%", low):
        if round(float(n), 1) not in facts_pct:
            return False, f"invents a percentage: {n}%"
    return True, "ok"


class MockLLM:
    """Stand-in for tests: 'good' follows the facts, 'invent' adds a pocket, 'fine' says all is well."""

    name = "mock"

    def __init__(self, mode="good"):
        self.mode = mode

    def __call__(self, prompt, facts):
        if self.mode == "invent":
            return template_advice(facts) + " Pocket 99 also looks damaged."
        if self.mode == "fine":
            return "The strip is fine, no action needed."
        return template_advice(facts)


class ExecuTorchLlama:
    """Calls the ExecuTorch llama_main runner. Flags follow the ExecuTorch 1.5 Raspberry Pi tutorial;
    confirm with `llama_main --help` on your build."""

    name = "llama"

    def __init__(self, runner, model, tokenizer, seq_len=128, timeout=60):
        self.cmd = [runner, "--model_path", model, "--tokenizer_path", tokenizer, "--seq_len", str(seq_len)]
        self.timeout = timeout

    def __call__(self, prompt, facts):
        out = subprocess.run(self.cmd + ["--prompt", prompt], capture_output=True, text=True, timeout=self.timeout)
        text = out.stdout
        if prompt in text:                      # the runner echoes the prompt before its answer
            text = text.split(prompt, 1)[1]
        return text.strip()


def advise(facts, llm=None):
    """Return {'text', 'source', 'check'}. source is none (PASS), llama, mock, or template."""
    if facts.get("verdict") != "FAIL":
        return {"text": "Strip passed. No action needed.", "source": "none", "check": "not needed"}
    if llm is not None:
        try:
            reply = llm(build_prompt(facts), facts)
            ok, why = fact_check(reply, facts)
            if ok:
                return {"text": reply, "source": getattr(llm, "name", "llm"), "check": "passed"}
            reason = f"LLM reply rejected: {why}"
        except Exception as e:  # runner missing, timeout, crash
            reason = f"LLM failed: {type(e).__name__}: {e}"
    else:
        reason = "no LLM configured"
    return {"text": template_advice(facts), "source": "template", "check": reason}
