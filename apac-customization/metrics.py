"""Shared scoring: confusion matrix, P/R/F1/FPR/MCC, and bootstrap 95% CIs.

Decoupled from running: every runner writes a per-row predictions CSV
(columns include at least `label` and `pred`, plus optional `lang`,`category`),
and this module scores any such file. Run `python metrics.py` to score all
`results/predictions_*.csv` into one consolidated table + `results/metrics_table.json`.

Metrics rationale (answers the "what does F1 mean here?" question):
 - F1 blends precision and recall equally; it can look high while the model is
   indiscriminately over-flagging (high recall, terrible precision). We therefore
   also report FPR (false-positive rate on legitimate content — the T&S cost that
   matters) and MCC (Matthews correlation), which is robust to class imbalance and
   near 0 for a model with no real discrimination.
 - Bootstrap 95% CIs quantify small-sample noise so "is this difference real?" is
   answerable instead of eyeballed.
 - These are verdict-emitting models with no tunable threshold, so every number is
   the model+policy *default operating point*, not a swept optimum.
"""
import csv
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# deterministic LCG so we never touch Math.random / time-seeding
_STATE = 0x2545F4914F6CDD1D


def _rand():
    global _STATE
    _STATE = (_STATE * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return _STATE / (1 << 64)


def confusion(pairs):
    tp = sum(1 for p, l in pairs if p == 1 and l == 1)
    fp = sum(1 for p, l in pairs if p == 1 and l == 0)
    fn = sum(1 for p, l in pairs if p != 1 and l == 1)
    tn = sum(1 for p, l in pairs if p != 1 and l == 0)
    return tp, fp, fn, tn


def scores(pairs):
    tp, fp, fn, tn = confusion(pairs)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    MCC = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return dict(n=len(pairs), pos=tp + fn, neg=tn + fp, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=P, recall=R, f1=F, fpr=FPR, mcc=MCC)


def bootstrap_ci(pairs, stat="f1", B=2000, alpha=0.05):
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(B):
        samp = [pairs[int(_rand() * n)] for _ in range(n)]
        vals.append(scores(samp)[stat])
    vals.sort()
    lo = vals[int((alpha / 2) * B)]
    hi = vals[int((1 - alpha / 2) * B)]
    return (round(lo, 3), round(hi, 3))


def load_pairs(path, lang=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if lang is not None and r.get("lang") != lang:
                continue
            try:
                l = int(r["label"])
            except (KeyError, ValueError):
                continue
            try:
                p = int(r["pred"])
            except (KeyError, ValueError):
                p = -1
            pairs.append((p, l))
    return pairs


def langs_in(path):
    with open(path, encoding="utf-8") as f:
        return sorted({r.get("lang", "") for r in csv.DictReader(f) if r.get("lang")})


def score_file(path, ci=True):
    pairs = load_pairs(path)
    s = scores(pairs)
    s["unparsed"] = sum(1 for p, l in pairs if p not in (0, 1))
    if ci and s["pos"] and s["neg"]:  # CIs only meaningful with both classes
        s["f1_ci"] = bootstrap_ci(pairs, "f1")
        s["mcc_ci"] = bootstrap_ci(pairs, "mcc")
    return s


def _row(name, s):
    ci = s.get("f1_ci", ("", ""))
    mci = s.get("mcc_ci", ("", ""))
    ci_s = f"[{ci[0]}-{ci[1]}]" if ci[0] != "" else "  (pos-only)"
    mci_s = f"[{mci[0]}-{mci[1]}]" if mci[0] != "" else ""
    print(f"{name:34} {s['n']:>4} {s['pos']:>4} {s['precision']:>5.2f} "
          f"{s['recall']:>5.2f} {s['f1']:>6.3f} {ci_s:>15} {s['fpr']:>5.2f} "
          f"{s['mcc']:>6.2f} {mci_s:>15} {s['unparsed']:>3}")


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "predictions_*.csv")))
    table = {}
    print(f"{'run [·lang]':34} {'n':>4} {'pos':>4} {'P':>5} {'R':>5} {'F1':>6} "
          f"{'F1 95% CI':>15} {'FPR':>5} {'MCC':>6} {'MCC 95% CI':>15} unp")
    print("-" * 120)
    for f in files:
        name = os.path.basename(f)[len("predictions_"):-len(".csv")]
        s = score_file(f)
        table[name] = s
        _row(name, s)
        # per-language breakdown when the file spans >1 language
        ls = langs_in(f)
        if len(ls) > 1:
            for L in ls:
                sl = scores(load_pairs(f, lang=L))
                sl["unparsed"] = sum(1 for p, l in load_pairs(f, lang=L) if p not in (0, 1))
                if sl["pos"] and sl["neg"]:
                    sl["f1_ci"] = bootstrap_ci(load_pairs(f, lang=L), "f1")
                    sl["mcc_ci"] = bootstrap_ci(load_pairs(f, lang=L), "mcc")
                table[f"{name}·{L}"] = sl
                _row(f"  {name}·{L}", sl)
    with open(os.path.join(HERE, "results", "metrics_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n-> results/metrics_table.json ({len(table)} rows)")


if __name__ == "__main__":
    main()
