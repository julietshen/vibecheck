"""Demo 4 — gpt-oss-safeguard-20b: a policy-steerable model that also SHOWS ITS
REASONING (auditability = the talk's accountability theme), tested for the APAC
language gap.

Same policies as the cope-b demo (fair head-to-head). On the SEA set we run the
moneylending steerability sweep; on Chinese (ChiFraud) and Bengali (TB-OLID) we run
one baseline policy and read per-language recall. We also print a couple of the
model's reasoning traces — the thing cope-b can't give you.

Served via vLLM on Modal (serve_safeguard.py); reasoning needs a big token budget.
Uses the TEST split only (cost control). From project root:
  VLLM_API_KEY=$(cat <scratch>/vllm_api_key) ../.venv/bin/python demo4_safeguard/run_safeguard.py
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
sys.path.insert(0, os.path.join(ROOT, "demo3_cope"))
from run_cope import POLICIES, OFFENSIVE_POLICIES  # identical policy text  # noqa: E402

ENDPOINT = "https://juliet--gpt-oss-safeguard-serve.modal.run/v1/chat/completions"
MODEL = "openai/gpt-oss-safeguard-20b"
API_KEY = os.environ.get("VLLM_API_KEY", "")

TEMPLATE = ("{policy}\n\nCONTENT TO CLASSIFY:\n{content}\n\n"
            "Answer with 1 if the content violates the policy, or 0 if it does not.")


def parse_last01(s):
    for ch in reversed((s or "").strip()):
        if ch in "01":
            return int(ch)
    return -1


def call(policy, content, keep_reason=False):
    for mt in (2048, 4096):  # retry with a bigger budget if the verdict is empty
        body = {"model": MODEL,
                "messages": [{"role": "user", "content": TEMPLATE.format(policy=policy, content=content)}],
                "max_tokens": mt, "temperature": 0.0, "top_p": 1.0}
        r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}",
                          "Content-Type": "application/json"}, json=body, timeout=600)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        msg = ch.get("message", {})
        content_out = msg.get("content") or ""
        reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
        v = parse_last01(content_out) if parse_last01(content_out) != -1 else parse_last01(reason)
        if v != -1:
            return (v, (reason or content_out)) if keep_reason else (v, None)
    return (-1, None)


def load_test(path):
    return [r for r in csv.DictReader(open(path)) if r["split"] == "test"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def run_set(name, path, policies, sweep=False):
    rows = load_test(path)
    langs = sorted(set(r["lang"] for r in rows))
    base_key = next(iter(policies))
    keys = list(policies) if sweep else [base_key]
    jobs = [(i, k) for i in range(len(rows)) for k in keys]
    preds = {k: [None] * len(rows) for k in keys}
    reasons = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {}
        for i, k in jobs:
            keep = (k == base_key and int(rows[i]["label"]) == 1 and rows[i]["lang"] not in reasons)
            futs[ex.submit(call, policies[k], rows[i]["text"], keep)] = (i, k, keep)
        for f in cf.as_completed(futs):
            i, k, keep = futs[f]
            try:
                v, reason = f.result()
            except Exception:
                v, reason = -1, None
            preds[k][i] = v
            if keep and reason and rows[i]["lang"] not in reasons:
                reasons[rows[i]["lang"]] = (rows[i]["text"][:80], reason.strip()[:300])

    labels = [int(r["label"]) for r in rows]
    pos = [i for i in range(len(rows)) if labels[i] == 1]
    base = preds[base_key]
    tp = sum(base[i] == 1 and labels[i] == 1 for i in range(len(rows)))
    fp = sum(base[i] == 1 and labels[i] == 0 for i in range(len(rows)))
    fn = sum(base[i] != 1 and labels[i] == 1 for i in range(len(rows)))
    p, rec, f1 = prf(tp, fp, fn)
    unparsed = sum(v == -1 for k in preds for v in preds[k])

    print(f"\n=== {name} ({len(rows)} test posts: {', '.join(langs)}) ===")
    print(f"[1] baseline '{base_key}':  P={p:.2f} R={rec:.2f} F1={f1:.3f}  (unparsed {unparsed})")
    for L in langs:
        lp = [i for i in pos if rows[i]["lang"] == L]
        print(f"    {L:3} positive recall {sum(base[i]==1 for i in lp)}/{len(lp)}")
    rel = None
    if sweep and all(k in preds for k in ("flag_illegal", "permit_all_loans", "flag_all_loans")):
        bf = sum(preds["flag_illegal"][i] == 1 for i in pos)
        released = sum(preds["flag_illegal"][i] == 1 and preds["permit_all_loans"][i] == 0 for i in pos)
        rel = released / bf if bf else 0
        print(f"[2] STEERABILITY release under 'permit_all_loans': {released}/{bf} ({rel:.0%})")
    return {"set": name, "n": len(rows), "langs": langs,
            "baseline": {"P": round(p, 3), "R": round(rec, 3), "F1": round(f1, 3)},
            "release_rate_permit_all": rel, "unparsed": unparsed,
            "reasoning_samples": reasons}


def main():
    if not API_KEY:
        sys.exit("set VLLM_API_KEY")
    # warmup
    print("Warming safeguard (cold start + reasoning)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 1500:
        try:
            call(next(iter(POLICIES.values())), "hello", False)
            print(f"  ready after {time.time()-t0:.0f}s", flush=True); break
        except Exception as e:
            print(f"  waiting ({type(e).__name__})...", flush=True); time.sleep(20)

    out = [run_set("SEA moneylending", os.path.join(ROOT, "data", "apac_moneylending.csv"),
                   POLICIES, sweep=True)]
    zh = os.path.join(ROOT, "data", "chifraud", "chifraud_sample.csv")
    if os.path.exists(zh):
        out.append(run_set("Chinese (ChiFraud)", zh, POLICIES))
    bn = os.path.join(ROOT, "data", "bengali", "bengali_offensive.csv")
    if os.path.exists(bn):
        out.append(run_set("Bengali offensive (TB-OLID)", bn, OFFENSIVE_POLICIES))

    # show a couple of reasoning traces (auditability)
    print("\n--- sample reasoning traces (auditable verdicts) ---")
    for r in out:
        for lang, (txt, reason) in (r.get("reasoning_samples") or {}).items():
            print(f"[{r['set']} · {lang}] post: {txt}\n   reasoning: {reason}\n")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary_demo4_safeguard.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("-> results/summary_demo4_safeguard.json")


if __name__ == "__main__":
    main()
