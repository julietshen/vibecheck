"""Shared before/after evaluation harness for the customization ladder.

Runs a (base | base+LoRA-adapter) MLX model over the APAC moneylending test set
and reports precision / recall / F1 overall, plus the two numbers that carry the
regional-customization story:

  - RECALL on illegal_ml  : did the model catch the region-specific harm?
  - FALSE-POSITIVE on licensed hard-negatives : did it over-flag legit lenders?
  - per-language recall on illegal_ml : where does a Western base model go blind?

Prompts are sent RAW (no chat template) so the base and the fine-tuned model are
compared on the identical string. Qwen3 "thinking" is not triggered because we
never apply the chat template; we read the first few generated tokens.

Usage:
  python run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --label base
  python run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --adapter rung2_lora/adapters --label tuned
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import chat_prompt, parse_verdict, POLICIES  # noqa: E402


def load_test(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-bf16")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--test", default=os.path.join(HERE, "data", "apac_moneylending.csv"))
    ap.add_argument("--label", required=True, help="name for the output artefact")
    ap.add_argument("--policy", choices=list(POLICIES), default="minimal")
    ap.add_argument("--max-tokens", type=int, default=6)
    args = ap.parse_args()
    policy = POLICIES[args.policy]

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    kw = {"adapter_path": args.adapter} if args.adapter else {}
    model, tok = load(args.model, **kw)
    sampler = make_sampler(temp=0.0)  # greedy, deterministic

    rows = load_test(args.test)
    preds = []
    for r in rows:
        out = generate(model, tok, prompt=chat_prompt(tok, r["text"], policy),
                       max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        preds.append({**r, "raw": out.strip(), "pred": parse_verdict(out)})

    # overall confusion (positive class = VIOLATION = 1)
    tp = sum(1 for r in preds if r["pred"] == 1 and int(r["label"]) == 1)
    fp = sum(1 for r in preds if r["pred"] == 1 and int(r["label"]) == 0)
    fn = sum(1 for r in preds if r["pred"] != 1 and int(r["label"]) == 1)
    tn = sum(1 for r in preds if r["pred"] != 1 and int(r["label"]) == 0)
    unparsed = sum(1 for r in preds if r["pred"] == -1)
    p, r_, f = prf(tp, fp, fn)

    # recall on illegal_ml per language
    langs = sorted(set(r["lang"] for r in preds))
    per_lang = {}
    for L in langs:
        pos = [r for r in preds if r["lang"] == L and int(r["label"]) == 1]
        caught = sum(1 for r in pos if r["pred"] == 1)
        per_lang[L] = (caught, len(pos))

    # negative-class false-positive rate. Prefer the "hard" negative category
    # (licensed lenders / legit finance) if the set marks one; else all negatives.
    hard = [r for r in preds if r["category"] in ("licensed", "normal")]
    negs = hard if hard else [r for r in preds if int(r["label"]) == 0]
    lic_fp = sum(1 for r in negs if r["pred"] == 1)

    summary = {
        "label": args.label, "model": args.model, "adapter": args.adapter,
        "policy": args.policy,
        "n": len(preds), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unparsed": unparsed,
        "precision": round(p, 3), "recall": round(r_, 3), "f1": round(f, 3),
        "illegal_recall_by_lang": {k: f"{v[0]}/{v[1]}" for k, v in per_lang.items()},
        "hardneg_false_positive": f"{lic_fp}/{len(negs)}",
    }

    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"predictions_{args.label}.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["text", "lang", "category", "label", "pred", "raw"])
        w.writeheader()
        for r in preds:
            w.writerow({k: r[k] for k in ("text", "lang", "category", "label", "pred", "raw")})
    with open(os.path.join(outdir, f"summary_{args.label}.json"), "w") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
