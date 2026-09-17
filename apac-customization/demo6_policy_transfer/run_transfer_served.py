"""Policy-language transfer on a SERVED steerable model (cope-b) — does the
"English policy wins" finding from Qwen generalize to a policy-following model?

Same design as run_transfer.py: hold content fixed, vary the policy language
(EN / TH / ZH), read F1. cope-b is the most steerable model in the suite, so it is
the cleanest test of whether policy LANGUAGE (not just content) matters.

Needs VLLM_API_KEY + the cope-b endpoint. From project root:
  VLLM_API_KEY=$(cat <scratch>/vllm_api_key) \
    ../.venv/bin/python demo6_policy_transfer/run_transfer_served.py
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
from common import POLICY_SCAM, POLICY_SCAM_TH, POLICY_SCAM_ZH  # noqa: E402

API_KEY = os.environ.get("VLLM_API_KEY", "")
ENDPOINT = "https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions"
MODEL = "zentropi-ai/cope-b-a4b"
TEMPLATE = ("INSTRUCTIONS\n============\n\nExamine the given POLICY and determine if the "
            "given CONTENT meets the criteria for ANY of the LABELS. Answer \"1\" if yes, "
            "and \"0\" if no.\n\n\nPOLICY\n======\n\n{policy}\n\n\nCONTENT\n=======\n\n"
            "{content}\n\n\nANSWER\n======\n\n")
# cope-b expects a LABEL:-prefixed policy; prepend one (keeps the in-language body).
_LABEL = "LABEL: scam. Flag CONTENT as a violation (1) if it matches the policy, else 0.\n\n"
POLICY_LANGS = {"EN": _LABEL + POLICY_SCAM, "TH": _LABEL + POLICY_SCAM_TH,
                "ZH": _LABEL + POLICY_SCAM_ZH}
CONTENT_SETS = {
    "Thai (tu_scam)": (os.path.join(ROOT, "data", "sea_scam", "sea_scam.csv"), "th"),
    "Chinese (ChiFraud)": (os.path.join(ROOT, "data", "chifraud", "chifraud_sample.csv"), "zh"),
    "English (moneylending)": (os.path.join(ROOT, "data", "apac_moneylending.csv"), "en"),
}


def call(policy, content):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": TEMPLATE.format(policy=policy, content=content)}],
            "max_tokens": 4, "temperature": 0.0, "top_p": 1.0}
    r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}",
                      "Content-Type": "application/json"}, json=body, timeout=600)
    r.raise_for_status()
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()
    if raw.startswith("1") or raw.startswith("VI"):
        return 1
    if raw.startswith("0") or raw.startswith("OK"):
        return 0
    return -1


def load(path, lang):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test" and r.get("lang") == lang]


def f1(rows, preds):
    tp = sum(preds[i] == 1 and int(rows[i]["label"]) == 1 for i in range(len(rows)))
    fp = sum(preds[i] == 1 and int(rows[i]["label"]) == 0 for i in range(len(rows)))
    fn = sum(preds[i] != 1 and int(rows[i]["label"]) == 1 for i in range(len(rows)))
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


def main():
    if not API_KEY:
        sys.exit("set VLLM_API_KEY")
    # warmup
    t0 = time.time()
    while time.time() - t0 < 1500:
        try:
            call(POLICY_SCAM, "hello"); break
        except Exception as e:
            print(f"warming cope-b ({type(e).__name__})...", flush=True); time.sleep(20)

    grid = {}
    for cname, (path, lang) in CONTENT_SETS.items():
        rows = load(path, lang)
        if not rows:
            continue
        grid[cname] = {}
        for pl, policy in POLICY_LANGS.items():
            preds = [None] * len(rows)
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                futs = {ex.submit(call, policy, rows[i]["text"]): i for i in range(len(rows))}
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    try:
                        preds[i] = fut.result()
                    except Exception:
                        preds[i] = -1
            f, p, rec = f1(rows, preds)
            grid[cname][pl] = {"F1": round(f, 3), "P": round(p, 2), "R": round(rec, 2), "n": len(rows)}

    print("\nPolicy-language x content-language transfer — cope-b-a4b (steerable)\n")
    header = "content \\ policy".ljust(26) + "".join(pl.ljust(16) for pl in POLICY_LANGS)
    print(header); print("-" * len(header))
    for cname, byp in grid.items():
        print(cname.ljust(26) + "".join(f"F1={byp[pl]['F1']}".ljust(16) for pl in POLICY_LANGS))

    with open(os.path.join(ROOT, "results", "summary_transfer_cope_b.json"), "w") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)
    print("\n-> results/summary_transfer_cope_b.json")


if __name__ == "__main__":
    main()
