"""Build a REAL Southeast-Asian scam/spam set to replace the synthetic SEA moneylending
set: Thai (tu_scam) + Tagalog (SPAM_SMS).

- Thai — tu_scam_dataset.xlsx: labels scam / normal / suspicious. We use a balanced
  binary scam(1) vs normal(0) set (train/valid/test); 'suspicious' is dropped from the
  binary metric (it's the ambiguous middle).
- Tagalog — SPAM_SMS.csv: real Philippine SMS spam, POSITIVE-ONLY (no benign labels), so
  Tagalog is evaluated recall-only (of real spam, how much does each model catch?).

Sources are the user's local files, NOT vendored. Point --thai / --tagalog at them:
  python data/build_sea_real.py \
     --thai ~/Downloads/tu_scam_dataset.csv.xlsx \
     --tagalog ~/Downloads/SPAM_SMS.csv

Writes data/sea_scam/sea_scam.csv (shared schema) + mlx/{train,valid,test}.jsonl for a
Thai fine-tune. Real messaging data — may contain scam/PII-shaped text.
"""
import argparse
import csv
import json
import os
import random
import sys

random.seed(20260916)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import build_prompt, label_word, POLICY_SCAM_MINIMAL  # noqa: E402
MAXLEN = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thai", required=True)
    ap.add_argument("--tagalog", required=True)
    ap.add_argument("--thai-train", type=int, default=200)   # per class
    ap.add_argument("--thai-test", type=int, default=150)    # per class (enlarged for tighter CIs)
    ap.add_argument("--tagalog-test", type=int, default=150)  # spam only (recall)
    args = ap.parse_args()

    import pandas as pd
    th = pd.read_excel(args.thai)
    scam = [str(t)[:MAXLEN] for t in th[th["label"] == "scam"]["text"].dropna()]
    normal = [str(t)[:MAXLEN] for t in th[th["label"] == "normal"]["text"].dropna()]
    random.shuffle(scam); random.shuffle(normal)
    ntr, nte = args.thai_train, args.thai_test
    th_train = [(1, t) for t in scam[:ntr]] + [(0, t) for t in normal[:ntr]]
    th_test = [(1, t) for t in scam[ntr:ntr + nte]] + [(0, t) for t in normal[ntr:ntr + nte]]
    random.shuffle(th_train); random.shuffle(th_test)
    th_valid = th_train[:30]; th_train = th_train[30:]

    sms = pd.read_csv(args.tagalog)
    tl_spam = [str(t)[:MAXLEN] for t in sms["text"].dropna() if len(str(t).strip()) > 8]
    random.shuffle(tl_spam)
    tl_test = [(1, t) for t in tl_spam[:args.tagalog_test]]

    outdir = os.path.join(HERE, "sea_scam")
    os.makedirs(os.path.join(outdir, "mlx"), exist_ok=True)
    with open(os.path.join(outdir, "sea_scam.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "lang", "category", "split"])
        w.writeheader()
        for split, rows, lang in [("train", th_train, "th"), ("valid", th_valid, "th"),
                                  ("test", th_test, "th"), ("test", tl_test, "tl")]:
            for label, text in rows:
                w.writerow({"text": text, "label": label, "lang": lang,
                            "category": "scam" if label else "normal", "split": split})
    # MLX data for a Thai scam fine-tune (Thai train/valid only)
    for name, rows in [("train", th_train), ("valid", th_valid), ("test", th_test)]:
        with open(os.path.join(outdir, "mlx", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for label, text in rows:
                f.write(json.dumps({"prompt": build_prompt(text, POLICY_SCAM_MINIMAL),
                                    "completion": " " + label_word(label)}, ensure_ascii=False) + "\n")

    print(f"Thai: train {len(th_train)} valid {len(th_valid)} test {len(th_test)} (balanced scam/normal)")
    print(f"Tagalog: test {len(tl_test)} (spam only, recall-eval)")
    print(f"-> {outdir}/sea_scam.csv + mlx/")


if __name__ == "__main__":
    main()
