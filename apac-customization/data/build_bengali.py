"""Build a small Bengali (romanized/code-mixed Bangla) offensive-language set from
TB-OLID for the language-coverage matrix.

TB-OLID = Transliterated Bangla Offensive Language ID (HASOC 2024 Bangla source),
AGPL-3.0. Real social-media Bangla, much of it romanized ("eta ki kuno date??") or
code-mixed with English. offensive_gold: O = offensive, N = not.

Source is NOT vendored (AGPL). Clone it, then point --source at the clone:
    git clone --depth 1 https://github.com/LanguageTechnologyLab/TB-OLID.git
    python data/build_bengali.py --source /path/to/TB-OLID

Writes data/bengali/bengali_offensive.csv (shared schema) + mlx/{train,valid,test}.jsonl
(for an optional Bengali LoRA). Domain = offensive; scored under the offensive policy.
NSFW: contains real offensive language by construction.
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
from common import build_prompt, label_word, POLICY_OFFENSIVE_MINIMAL  # noqa: E402

MAXLEN = 300


def load(path, n_pos, n_neg):
    data = json.load(open(path))
    pos = [r["text"].strip() for r in data if r["offensive_gold"] == "O" and r["text"].strip()]
    neg = [r["text"].strip() for r in data if r["offensive_gold"] == "N" and r["text"].strip()]
    random.shuffle(pos); random.shuffle(neg)
    rows = [(1, t[:MAXLEN]) for t in pos[:n_pos]] + [(0, t[:MAXLEN]) for t in neg[:n_neg]]
    random.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to a TB-OLID clone")
    args = ap.parse_args()

    train = load(os.path.join(args.source, "train.json"), 150, 150)
    test = load(os.path.join(args.source, "test.json"), 150, 150)  # enlarged for tighter CIs
    valid = train[:30]; train = train[30:]

    outdir = os.path.join(HERE, "bengali")
    os.makedirs(os.path.join(outdir, "mlx"), exist_ok=True)
    with open(os.path.join(outdir, "bengali_offensive.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "lang", "category", "split"])
        w.writeheader()
        for name, rows in [("train", train), ("valid", valid), ("test", test)]:
            for label, text in rows:
                w.writerow({"text": text, "label": label, "lang": "bn",
                            "category": "offensive" if label == 1 else "benign", "split": name})
    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        with open(os.path.join(outdir, "mlx", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for label, text in rows:
                f.write(json.dumps({"prompt": build_prompt(text, POLICY_OFFENSIVE_MINIMAL),
                                    "completion": " " + label_word(label)}, ensure_ascii=False) + "\n")
    print(f"train {len(train)} valid {len(valid)} test {len(test)} -> {outdir}/")


if __name__ == "__main__":
    main()
