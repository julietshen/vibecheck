"""Build a cross-lingual gendered-abuse set: Chinese (SWSR) + Hindi/Tamil/Indian
English (Uli). Lets the matrix cover gendered abuse across languages incl. Chinese.

- SWSR (Sina Weibo Sexism Review, aggiejiang/SWSR) — Chinese, `comment_text` + `label`
  (1 sexist / 0 not).
- Uli (tattle-made/uli_dataset, CC BY 4.0) — Hindi/Tamil/Indian English, question_1 =
  "is this gendered abuse when directed at a person of marginalized gender". Labels are
  per-annotator columns; we take the majority of non-empty votes.

Sources NOT vendored. Clone them, then:
  python data/build_gendered.py --swsr /path/to/SWSR --uli /path/to/uli_dataset

Writes data/gendered/gendered_abuse.csv (shared schema). NSFW: real abusive content.
"""
import argparse
import csv
import os
import random
import sys

random.seed(20260916)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
MAXLEN = 300


def sample(rows, n_pos, n_neg):
    pos = [t for t, l in rows if l == 1]
    neg = [t for t, l in rows if l == 0]
    random.shuffle(pos); random.shuffle(neg)
    out = [(t, 1) for t in pos[:n_pos]] + [(t, 0) for t in neg[:n_neg]]
    random.shuffle(out)
    return out


def load_swsr(path):
    rows = []
    with open(os.path.join(path, "SWSR", "SexComment.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = (r.get("comment_text") or "").strip()
            lab = (r.get("label") or "").strip()
            if t and lab in ("0", "1"):
                rows.append((t[:MAXLEN], int(lab)))
    return rows


def load_uli(path, lang):
    """Majority vote over non-empty annotator columns of train_{lang}_l1.csv."""
    rows = []
    fp = os.path.join(path, "training", f"train_{lang}_l1.csv")
    with open(fp, encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        ann_cols = [i for i, h in enumerate(header) if h not in ("text", "key")]
        ti = header.index("text")
        for rec in rdr:
            if len(rec) <= max(ann_cols + [ti]):
                continue
            votes = []
            for i in ann_cols:
                v = rec[i].strip()
                if v not in ("", "NA"):
                    try:
                        votes.append(1 if float(v) >= 0.5 else 0)
                    except ValueError:
                        pass
            t = rec[ti].strip()
            if t and votes:
                rows.append((t[:MAXLEN], 1 if sum(votes) * 2 >= len(votes) and sum(votes) > 0 else 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swsr", required=True)
    ap.add_argument("--uli", required=True)
    ap.add_argument("--per-lang", type=int, default=40, help="pos+neg per language for test")
    args = ap.parse_args()

    n = args.per_lang // 2
    datasets = {"zh": sample(load_swsr(args.swsr), n, n)}
    for lang in ("en", "hi", "ta"):
        datasets[lang] = sample(load_uli(args.uli, lang), n, n)

    outdir = os.path.join(HERE, "gendered")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gendered_abuse.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "lang", "category", "split"])
        w.writeheader()
        for lang, rows in datasets.items():
            for text, label in rows:
                w.writerow({"text": text, "label": label, "lang": lang,
                            "category": "gendered_abuse" if label else "benign", "split": "test"})
    for lang, rows in datasets.items():
        pos = sum(1 for _, l in rows if l == 1)
        print(f"{lang}: {len(rows)} rows ({pos} abuse / {len(rows)-pos} not)")
    print(f"-> {outdir}/gendered_abuse.csv")


if __name__ == "__main__":
    main()
