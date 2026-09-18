"""Served policy-following models (cope-b, safeguard) on ALL enlarged test sets, saving
per-row predictions + latency for metrics.py. One H100 session covers both models × three
datasets (Thai/PH scam, Chinese fraud, Bengali offensive).

Each dataset uses the same generic policy the local Qwen runs used, wrapped in cope's
LABEL:-prefixed template. Needs VLLM_API_KEY. From project root:
  VLLM_API_KEY=$(cat <scratch>/vllm_api_key) ../.venv/bin/python demo3_cope/run_served_matrix.py
"""
import concurrent.futures as cf
import csv
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import (POLICY_SCAM, POLICY_MINIMAL, POLICY_OFFENSIVE)  # noqa: E402

API_KEY = os.environ.get("VLLM_API_KEY", "")
TEMPLATE = ("INSTRUCTIONS\n============\n\nExamine the given POLICY and determine if the "
            "given CONTENT meets the criteria for ANY of the LABELS. Answer \"1\" if yes, "
            "and \"0\" if no.\n\n\nPOLICY\n======\n\n{policy}\n\n\nCONTENT\n=======\n\n"
            "{content}\n\n\nANSWER\n======\n\n")
_LABEL = "LABEL: violation. Flag CONTENT as a violation (1) if it matches the policy, else 0.\n\n"

MODELS = {
    "cope-b": ("https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions",
               "zentropi-ai/cope-b-a4b", 4),
    "safeguard": ("https://juliet--gpt-oss-safeguard-serve.modal.run/v1/chat/completions",
                  "openai/gpt-oss-safeguard-20b", 2048),
}
DATASETS = {
    "sea_scam": (os.path.join(ROOT, "data", "sea_scam", "sea_scam.csv"), _LABEL + POLICY_SCAM),
    "chifraud": (os.path.join(ROOT, "data", "chifraud", "chifraud_sample.csv"), _LABEL + POLICY_MINIMAL),
    "bengali": (os.path.join(ROOT, "data", "bengali", "bengali_offensive.csv"), _LABEL + POLICY_OFFENSIVE),
}


def parse(raw):
    r = (raw or "").strip().upper()
    if r.startswith("1") or r.startswith("VI"):
        return 1
    if r.startswith("0") or r.startswith("OK"):
        return 0
    for ch in reversed(r):
        if ch in "01":
            return int(ch)
    return -1


def call(ep, model, mt, policy, content):
    body = {"model": model,
            "messages": [{"role": "user", "content": TEMPLATE.format(policy=policy, content=content)}],
            "max_tokens": mt, "temperature": 0.0, "top_p": 1.0}
    r = requests.post(ep, headers={"Authorization": f"Bearer {API_KEY}",
                      "Content-Type": "application/json"}, json=body, timeout=600)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    v = parse(msg.get("content") or "")
    if v == -1:
        v = parse(msg.get("reasoning_content") or msg.get("reasoning") or "")
    return v


def load_test(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def warmup(ep, model, mt, policy):
    t0 = time.time()
    while time.time() - t0 < 1500:
        try:
            call(ep, model, mt, policy, "hello"); print(f"  {model} ready ({time.time()-t0:.0f}s)", flush=True); return True
        except Exception as e:
            print(f"  warming {model} ({type(e).__name__})...", flush=True); time.sleep(20)
    return False


def main():
    if not API_KEY:
        sys.exit("set VLLM_API_KEY")
    summary = {}
    for mname, (ep, model, mt) in MODELS.items():
        print(f"\n===== {mname} =====", flush=True)
        first_policy = _LABEL + POLICY_SCAM
        if not warmup(ep, model, mt, first_policy):
            print(f"  {mname} did not warm; skipping"); continue
        for dname, (path, policy) in DATASETS.items():
            rows = load_test(path)
            preds = [None] * len(rows); lat = [None] * len(rows)

            def timed(i):
                t0 = time.perf_counter()
                v = call(ep, model, mt, policy, rows[i]["text"])
                return i, v, (time.perf_counter() - t0) * 1000
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                for fut in cf.as_completed([ex.submit(timed, i) for i in range(len(rows))]):
                    try:
                        i, v, ms = fut.result(); preds[i] = v; lat[i] = ms
                    except Exception:
                        pass
            preds = [p if p is not None else -1 for p in preds]
            lats = sorted(x for x in lat if x is not None)
            median_ms = round(lats[len(lats) // 2], 1) if lats else 0
            with open(os.path.join(ROOT, "results", f"predictions_served_{mname}_{dname}.csv"),
                      "w", newline="", encoding="utf-8") as pf:
                w = csv.writer(pf); w.writerow(["lang", "category", "label", "pred"])
                for i, r in enumerate(rows):
                    w.writerow([r.get("lang", ""), r.get("category", ""), r["label"], preds[i]])
            unparsed = sum(1 for p in preds if p == -1)
            summary[f"{mname}_{dname}"] = {"n": len(rows), "median_ms": median_ms, "unparsed": unparsed}
            print(f"  {dname:10} n={len(rows)} median={median_ms}ms unparsed={unparsed}", flush=True)

    with open(os.path.join(ROOT, "results", "summary_served_matrix.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n-> results/summary_served_matrix.json + predictions_served_*.csv")


if __name__ == "__main__":
    main()
