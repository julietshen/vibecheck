"""vibecheck — policy-conditioned safety-classifier evaluation harness.

One shared core; per-model adapters under models/. An adapter is a module with:

    NAME                  display name (str)
    SUPPORTS_CONCURRENCY  True for HTTP-backed models, False for local inference
    make_classifier(opts: dict) -> classify(policy_text, content, attempt=0)
                          returning (pred, raw): pred in {"1","0",""},
                          raw is the model's raw output, or "score=<0-1>" for
                          score-based models. attempt>0 signals a retry, so the
                          adapter may raise its output budget.

Usage:
    python eval.py --model shieldstral --label sh \
        --policies minimal simple medium full very_long zentropi_official
    python eval.py --model cope_b --test-set sexual_content_eval/test_set.csv --label sex \
        --policies sexual_content_minimal ... --concurrency 16
    python eval.py --model safeguard --probes probes/selfharm.csv ...

Everything model-agnostic lives here: test-set schema (id, content,
ground_truth), policy loading, per-policy checkpointing (resume after
interrupt), one retry on empty responses, optional calibration probes,
metrics with explicit error accounting, and the predictions/summary CSV
format shared by every run in results/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
POLICIES_DIR = HERE / "policies"
RESULTS_DIR = HERE / "results"
DEFAULT_TEST_SET = HERE / "test_set.csv"


def load_policies(names: list[str]) -> dict[str, str]:
    out = {}
    for name in names:
        path = POLICIES_DIR / f"{name}.md"
        if not path.exists():
            sys.exit(f"missing policy file: {path}")
        out[name] = path.read_text()
    return out


def load_test_set(path: Path, limit: int | None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("ground_truth", "").strip() in ("0", "1")]
    return rows[:limit] if limit else rows


def metrics(preds: list[str], truth: list[str]) -> dict:
    tp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "1")
    fp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "0")
    fn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "1")
    tn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "0")
    errors = sum(1 for p in preds if p == "")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(preds) - errors, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, errors=errors,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def run_probes(classify, probes_path: Path, policy_name: str, policy_text: str) -> None:
    """Calibration probes: cheap insurance against an inverted or broken setup
    before burning a full sweep. Warns loudly; aborts only on a fully inverted
    result (every probe wrong)."""
    probes = load_test_set(probes_path, None)
    print(f"calibration probes ({probes_path.name}) against policy '{policy_name}':")
    wrong = 0
    for p in probes:
        pred, raw = classify(policy_text, p["content"])
        ok = pred == p["ground_truth"]
        wrong += not ok
        marker = "ok" if ok else "MISMATCH"
        print(f"  [{marker}] expected={p['ground_truth']} got={pred!r} raw={raw[:60]!r} | {p['content'][:60]}")
    if probes and wrong == len(probes):
        sys.exit("every probe came out wrong — the score orientation or prompt "
                 "template is likely inverted/broken. Aborting before the full run.")
    if wrong:
        print(f"  WARNING: {wrong}/{len(probes)} probes mismatched — inspect before trusting the sweep.")


def classify_with_retry(classify, policy_text: str, content: str) -> tuple[str, str]:
    """One retry on an empty prediction (e.g. reasoning-budget truncation
    returning an empty final channel — see RESULTS.md on gpt-oss-safeguard).
    The attempt index is passed to the adapter so it can raise its budget."""
    try:
        pred, raw = classify(policy_text, content, 0)
    except Exception as e:
        pred, raw = "", f"ERROR: {e}"
    if pred == "":
        try:
            pred, raw = classify(policy_text, content, 1)
            if pred != "":
                raw = f"(retry) {raw}"
        except Exception as e:
            raw = f"ERROR on retry: {e}"
    return pred, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="adapter module name under models/ (e.g. shieldstral, cope_b)")
    ap.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    ap.add_argument("--label", default=None, help="tag for output filenames (e.g. sh, sex)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel requests (HTTP adapters only; local adapters run sequentially)")
    ap.add_argument("--probes", type=Path, default=None,
                    help="optional calibration CSV (id, content, ground_truth) run before the sweep")
    ap.add_argument("--model-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="adapter option, repeatable (e.g. --model-arg endpoint=https://... --model-arg max_tokens=4096)")
    args = ap.parse_args()

    opts = {}
    for kv in args.model_arg:
        k, _, v = kv.partition("=")
        opts[k] = v

    sys.path.insert(0, str(HERE))
    mod = importlib.import_module(f"models.{args.model}")
    classify = mod.make_classifier(opts)
    concurrency = args.concurrency if getattr(mod, "SUPPORTS_CONCURRENCY", False) else 1

    policies = load_policies(args.policies)
    rows = load_test_set(args.test_set, args.limit)
    print(f"model={getattr(mod, 'NAME', args.model)}  rows={len(rows)}  "
          f"policies={list(policies)}  concurrency={concurrency}")

    if args.probes:
        first = next(iter(policies))
        run_probes(classify, args.probes, first, policies[first])

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = args.label or "run"
    results: dict[tuple[str, str], tuple[str, str]] = {}
    ckpts = []
    for pname, ptext in policies.items():
        ckpt = RESULTS_DIR / f"ckpt_{args.model}_{tag}_{pname}.csv"
        ckpts.append(ckpt)
        done: dict[str, tuple[str, str]] = {}
        if ckpt.exists():
            with open(ckpt) as f:
                done = {r[0]: (r[1], r[2]) for r in csv.reader(f)}
            print(f"[{pname}] resuming, {len(done)} already scored")
        todo = [r for r in rows if r["id"] not in done]
        for r in rows:
            if r["id"] in done:
                results[(r["id"], pname)] = done[r["id"]]

        f = open(ckpt, "a", newline="")
        w = csv.writer(f)
        t0 = time.time()
        completed = 0

        def handle(row_id: str, pred: str, raw: str):
            nonlocal completed
            results[(row_id, pname)] = (pred, raw)
            w.writerow([row_id, pred, raw])
            f.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(todo):
                rate = completed / max(time.time() - t0, 1e-9)
                print(f"[{pname}] {len(done)+completed}/{len(rows)} "
                      f"({rate:.2f}/s, eta {(len(todo)-completed)/max(rate,1e-9)/60:.0f} min)", flush=True)

        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(classify_with_retry, classify, ptext, r["content"]): r["id"] for r in todo}
                for fut in as_completed(futs):
                    pred, raw = fut.result()
                    handle(futs[fut], pred, raw)
        else:
            for r in todo:
                pred, raw = classify_with_retry(classify, ptext, r["content"])
                handle(r["id"], pred, raw)
        f.close()
        print(f"[{pname}] complete", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.model}_{tag}_{ts}"
    policy_names = list(policies)

    pred_path = RESULTS_DIR / f"predictions_{suffix}.csv"
    header = ["id", "content", "ground_truth"]
    for p in policy_names:
        header += [f"{p}_pred", f"{p}_raw"]
    header += [f"{p}_correct" for p in policy_names]
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            line = [row["id"], row["content"], row["ground_truth"]]
            for p in policy_names:
                pred, raw = results[(row["id"], p)]
                line += [pred, raw]
            for p in policy_names:
                pred, _ = results[(row["id"], p)]
                line.append("1" if pred == row["ground_truth"] else "0")
            w.writerow(line)
    print(f"\nwrote predictions: {pred_path}")

    summary_path = RESULTS_DIR / f"summary_{suffix}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n", "tp", "fp", "fn", "tn", "errors",
                    "precision", "recall", "f1", "accuracy"])
        print(f"\n{'policy':36s} {'n':>4s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'tn':>4s} "
              f"{'err':>4s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'acc':>6s}")
        print("-" * 100)
        for p in policy_names:
            preds = [results[(row["id"], p)][0] for row in rows]
            truth = [row["ground_truth"] for row in rows]
            m = metrics(preds, truth)
            w.writerow([p, len(rows), m["tp"], m["fp"], m["fn"], m["tn"], m["errors"],
                        f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                        f"{m['f1']:.3f}", f"{m['accuracy']:.3f}"])
            print(f"{p:36s} {len(rows):>4d} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d} {m['tn']:>4d} "
                  f"{m['errors']:>4d} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['accuracy']:>6.3f}")
    print(f"wrote summary: {summary_path}")

    for ckpt in ckpts:
        ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
