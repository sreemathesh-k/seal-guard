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
            "\nFirst say the strip must be rejected. Then write one short sentence for each failed pocket, "
            "naming the pocket number, its defect and its severity exactly as given. "
            "End with one practical check for the line.")


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
    mentioned |= {int(n) for grp in re.findall(r"pockets?\s+((?:\d+\s*(?:,\s*and|,|and|&)\s*)+\d+)", low) for n in re.findall(r"\d+", grp)}
    if mentioned - allowed:
        return False, f"mentions pockets not in the facts: {sorted(mentioned - allowed)}"
    if allowed - mentioned:
        return False, f"leaves out failed pockets: {sorted(allowed - mentioned)}"
    sev = {int(p["pocket"]): (p.get("severity") or "").lower() for p in bad}
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", low):
        said = set(re.findall(r"\b(critical|major|minor)\b", sent))
        if len(said) != 1:
            continue
        s1 = said.pop()
        nums = {int(n) for n in re.findall(r"pockets?\s*(?:no\.?|number|#)?\s*(\d+)", sent)}
        nums |= {int(n) for g in re.findall(r"pockets?\s+((?:\d+\s*(?:,\s*and|,|and|&)\s*)+\d+)", sent) for n in re.findall(r"\d+", g)}
        wrong = sorted(n for n in nums if sev.get(n) and sev[n] != s1)
        if wrong:
            return False, f"states severity {s1} for pockets {wrong}, the facts disagree"
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
    """Runs Llama 3.2 1B Instruct through the ExecuTorch llama_main runner.
    Flags confirmed with `llama_main --help` on our ExecuTorch release/1.5 build.
    Wraps the prompt in the Llama 3 chat format, keeps only the assistant reply,
    and keeps the runner's own timing line (PyTorchObserver) in self.stats."""

    name = "llama"
    SYSTEM = ("You are a quality-control assistant for blister-pack inspection. "
              "Use only the facts given. Do not invent pocket numbers, values or defect types.")

    def __init__(self, runner, model, tokenizer, seq_len=128, timeout=120, max_new_tokens=120):
        # seq_len is kept so older callers still work; the runner marks it deprecated,
        # so we cap the reply with --max_new_tokens instead.
        self.cmd = [runner, "--model_path", model, "--tokenizer_path", tokenizer,
                    "--max_new_tokens", str(max_new_tokens), "--temperature", "0"]
        self.timeout = timeout
        self.stats = None

    def chat(self, prompt):
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + self.SYSTEM +
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + prompt +
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    def __call__(self, prompt, facts):
        import json
        out = subprocess.run(self.cmd + ["--prompt", self.chat(prompt)],
                             capture_output=True, text=True, timeout=self.timeout)
        text = out.stdout + "\n" + out.stderr
        for line in text.splitlines():
            if line.startswith("PyTorchObserver "):
                try:
                    self.stats = json.loads(line.split(" ", 1)[1])
                except ValueError:
                    pass
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker not in text:
            return ""                     # no reply found -> advise() falls back to the template
        text = text.rsplit(marker, 1)[1]
        text = text.split("<|eot_id|>", 1)[0].split("PyTorchObserver", 1)[0]
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
