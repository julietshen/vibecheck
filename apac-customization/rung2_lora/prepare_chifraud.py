"""Build a small REAL-data fine-tuning set from ChiFraud (Chinese fraud detection).

ChiFraud (Tang et al., COLING 2025) is a Chinese fraud-text benchmark, CC BY-NC 4.0.
We use two of its 11 classes to mirror the synthetic loan-shark demo on real data:

    label 9  地下黑贷  (underground / illegal loans)  -> VIOLATION  (positive)
    label 0  正常      (normal text, incl. legit finance) -> OK      (negative)

The "normal" class contains legitimate banking / finance text, so it supplies the
same hard-negative structure as the synthetic set: the model must separate an
illegal underground-loan ad from ordinary financial writing, in Chinese.

Source is NOT vendored (94 MB, CC BY-NC). Clone it yourself, then point --source
at the clone:
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/xuemingxxx/ChiFraud.git
    python rung2_lora/prepare_chifraud.py --source /path/to/ChiFraud

Writes data/chifraud/{train,valid,test}.jsonl (MLX) + chifraud_sample.csv (eval).
The temporal split t2023 is used for test (distribution-shift eval, per the paper).
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
from common import build_prompt, label_word, POLICY_MINIMAL  # noqa: E402

MAXLEN = 320  # truncate long Chinese paragraphs for fast LoRA on a laptop


def read_tsv(path, want_labels):
    """ChiFraud files are tab-separated with a Label_id/Text header (.csv ext)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        next(rdr, None)  # header
        for rec in rdr:
            if len(rec) < 2:
                continue
            lab, text = rec[0].strip(), rec[1].strip()
            if lab in want_labels and text:
                rows.append((lab, text[:MAXLEN]))
    return rows


def sample_split(source, split_file, n_pos, n_neg):
    rows = read_tsv(os.path.join(source, "dataset", split_file), {"9", "0"})
    pos = [r for r in rows if r[0] == "9"]
    neg = [r for r in rows if r[0] == "0"]
    random.shuffle(pos); random.shuffle(neg)
    out = [(1, t) for _, t in pos[:n_pos]] + [(0, t) for _, t in neg[:n_neg]]
    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to a ChiFraud clone")
    ap.add_argument("--train-pos", type=int, default=150)
    ap.add_argument("--train-neg", type=int, default=150)
    ap.add_argument("--test-pos", type=int, default=40)
    ap.add_argument("--test-neg", type=int, default=40)
    args = ap.parse_args()

    train = sample_split(args.source, "ChiFraud_train.csv", args.train_pos, args.train_neg)
    # valid: a small slice carved from train file (disjoint via ordering)
    valid = sample_split(args.source, "ChiFraud_t2022.csv", 15, 15)
    # test: temporal-shift split (2023), the paper's hard evaluation
    test = sample_split(args.source, "ChiFraud_t2023.csv", args.test_pos, args.test_neg)

    outdir = os.path.join(ROOT, "data", "chifraud")
    os.makedirs(outdir, exist_ok=True)

    # MLX jsonl (prompt/completion, minimal English policy — a non-regional team)
    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        with open(os.path.join(outdir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for label, text in rows:
                f.write(json.dumps({
                    "prompt": build_prompt(text, POLICY_MINIMAL),
                    "completion": " " + label_word(label),
                }, ensure_ascii=False) + "\n")

    # eval CSV in the shared schema (text,label,lang,category,split)
    with open(os.path.join(outdir, "chifraud_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "lang", "category", "split"])
        w.writeheader()
        for name, rows in [("train", train), ("valid", valid), ("test", test)]:
            for label, text in rows:
                w.writerow({"text": text, "label": label, "lang": "zh",
                            "category": "underground_loan" if label == 1 else "normal",
                            "split": name})

    print(f"train {len(train)}  valid {len(valid)}  test {len(test)}")
    print(f"-> {outdir}/(train|valid|test).jsonl + chifraud_sample.csv")


if __name__ == "__main__":
    main()
