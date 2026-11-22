"""Tamper-evident inspection log: a CSV where each row stores the SHA-256 hash of
(previous row's hash + this row's content). Editing, deleting or reordering any
earlier row breaks the chain, and verify() reports the first broken row.
"""
import csv
import hashlib
import json
import os
import time

FIELDS = ["time", "verdict", "facts", "advice", "source", "prev_hash", "hash"]
GENESIS = "0" * 64


def _digest(prev, row):
    body = "|".join([prev, row["time"], row["verdict"], row["facts"], row["advice"], row["source"]])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append(path, facts, advice, source):
    rows = _rows(path)
    prev = rows[-1]["hash"] if rows else GENESIS
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": str(facts.get("verdict", "")),
           "facts": json.dumps(facts, sort_keys=True, separators=(",", ":")),
           "advice": advice, "source": source, "prev_hash": prev}
    row["hash"] = _digest(prev, row)
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row["hash"]


def verify(path):
    """Return (ok, message)."""
    prev = GENESIS
    rows = _rows(path)
    for i, row in enumerate(rows, start=1):
        if row["prev_hash"] != prev or _digest(prev, row) != row["hash"]:
            return False, f"TAMPERED: chain breaks at row {i}"
        prev = row["hash"]
    return True, f"OK: {len(rows)} rows, chain intact"
