"""Q1 + Q2 — Policy-language x content-language transfer grid.

Does an ENGLISH policy work on APAC-language content? Does an APAC-language policy work
on that language AND on English content? We hold the CONTENT fixed and vary only the
LANGUAGE the (same) scam policy is written in — English / Thai / Chinese — and read F1.

Content sets (test split):
  - Thai scam        (tu_scam)         -> policies EN, TH, ZH
  - Chinese fraud    (ChiFraud)        -> policies EN, TH, ZH
  - English content  (SEA moneylending, English rows, as an English scam/benign proxy)

If F1 is stable across policy languages, the model transfers the policy across the
language barrier (you can write policy in English and moderate Thai). If it drops when
the policy language != content language, policy language matters.

Runs locally on Qwen3 via MLX (free). From project root:
  ../.venv/bin/python demo6_policy_transfer/run_transfer.py --model mlx-community/Qwen3-1.7B-bf16
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import (POLICY_SCAM, POLICY_SCAM_TH, POLICY_SCAM_ZH,  # noqa: E402
                    chat_prompt, parse_verdict)

POLICY_LANGS = {"EN": POLICY_SCAM, "TH": POLICY_SCAM_TH, "ZH": POLICY_SCAM_ZH}

CONTENT_SETS = {
    "Thai (tu_scam)": (os.path.join(ROOT, "data", "sea_scam", "sea_scam.csv"), "th"),
    "Chinese (ChiFraud)": (os.path.join(ROOT, "data", "chifraud", "chifraud_sample.csv"), "zh"),
    "English (moneylending)": (os.path.join(ROOT, "data", "apac_moneylending.csv"), "en"),
}


def load(path, lang):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == "test" and r.get("lang", lang) == lang]
    return rows


def f1(rows, preds):
    tp = sum(preds[i] == 1 and int(rows[i]["label"]) == 1 for i in range(len(rows)))
    fp = sum(preds[i] == 1 and int(rows[i]["label"]) == 0 for i in range(len(rows)))
    fn = sum(preds[i] != 1 and int(rows[i]["label"]) == 1 for i in range(len(rows)))
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-bf16")
    args = ap.parse_args()
    from mlx_lm import load as mlx_load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = mlx_load(args.model)
    sampler = make_sampler(temp=0.0)

    grid = {}
    for cname, (path, lang) in CONTENT_SETS.items():
        rows = load(path, lang)
        if not rows:
            continue
        grid[cname] = {}
        for pl, policy in POLICY_LANGS.items():
            preds = []
            for r in rows:
                out = generate(model, tok, prompt=chat_prompt(tok, r["text"], policy),
                               max_tokens=6, sampler=sampler, verbose=False)
                preds.append(parse_verdict(out))
            f, p, rec = f1(rows, preds)
            grid[cname][pl] = {"F1": round(f, 3), "P": round(p, 2), "R": round(rec, 2), "n": len(rows)}

    # print grid
    print(f"\nPolicy-language x content-language transfer — {args.model}\n")
    header = "content \\ policy".ljust(26) + "".join(pl.ljust(18) for pl in POLICY_LANGS)
    print(header); print("-" * len(header))
    for cname, byp in grid.items():
        row = cname.ljust(26)
        for pl in POLICY_LANGS:
            c = byp.get(pl, {})
            row += f"F1={c.get('F1','—')}".ljust(18)
        print(row)
    print("\n(EN column = English policy on that content; TH/ZH = in-language policy.)")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = args.model.split("/")[-1]
    with open(os.path.join(ROOT, "results", f"summary_transfer_{tag}.json"), "w") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)
    print(f"-> results/summary_transfer_{tag}.json")


if __name__ == "__main__":
    main()
