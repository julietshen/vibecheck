"""Rung 3 — a purpose-built, multilingual guard model, off the shelf (no training).

Runs Qwen/Qwen3Guard-Gen-8B (already cached locally) over our test posts. This is
a dedicated safety model in the Qwen family: it applies its OWN fixed safety
taxonomy across 119 languages with ZERO policy customization — you send content,
it returns Safety: Safe|Unsafe|Controversial.

The trade-off this illustrates (the top rung of the ladder):
  + strong multilingual coverage and no training / no policy authoring;
  - a FIXED taxonomy — you cannot express the licensed-vs-illegal nuance or a
    local carve-out the way rung-1 policies or a rung-2 fine-tune can.

For the SEA-specialized equivalent, swap the model id for
aisingapore/Qwen-SEA-Guard-8B-2602 (SEA-Guard, fine-tuned from SEA-LION v4 for
Southeast Asian cultural norms; native id/ms/my/ta/th/tl/vi/en). Same interface.

Run from project root (torch/MPS, ~8B; slower than the MLX rungs):
  ../.venv/bin/python rung3_regional/run_guard.py --limit 20
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODEL = "Qwen/Qwen3Guard-Gen-8B"
SAFE_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
CAT_RE = re.compile(r"Categories?:\s*(.+)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--test", default=os.path.join(ROOT, "data", "apac_moneylending.csv"))
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [r for r in csv.DictReader(open(args.test)) if r["split"] == "test"][:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="mps")

    tp = fp = fn = tn = 0
    print(f"\nRung 3 — {args.model} (fixed taxonomy, {len(rows)} multilingual posts)\n")
    print("lang  gold        guard-label   category")
    print("-" * 64)
    for r in rows:
        messages = [{"role": "user", "content": r["text"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
        gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        m = SAFE_RE.search(gen)
        label = m.group(1).capitalize() if m else "?"
        cat = (CAT_RE.search(gen).group(1).strip()[:24] if CAT_RE.search(gen) else "")
        pred = 1 if label in ("Unsafe", "Controversial") else 0
        gold = int(r["label"])
        tp += pred == 1 and gold == 1
        fp += pred == 1 and gold == 0
        fn += pred == 0 and gold == 1
        tn += pred == 0 and gold == 0
        gold_s = "VIOLATION" if gold else "OK"
        print(f"{r['lang']:5} {gold_s:11} {label:13} {cat}")

    p = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0
    print("-" * 64)
    print(f"vs our labels: P={p:.2f} R={rec:.2f} F1={f1:.3f}  (tp{tp} fp{fp} fn{fn} tn{tn})")
    print("\nNote: Qwen3Guard scores content against its own fixed safety taxonomy,")
    print("not our moneylending policy — it has no notion of 'licensed lender = OK'.")
    print("That fixed taxonomy is the top-rung trade-off: broadest coverage, least")
    print("configurable. SEA-Guard specializes the SAME approach for SEA norms.\n")


if __name__ == "__main__":
    main()
