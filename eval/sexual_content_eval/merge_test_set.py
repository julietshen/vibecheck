"""
Merge labelled Bluesky candidates + the synthetic red-team set into one test CSV
for the cope sexual-content eval.

Reads:
  - candidates_to_label.csv (manually labelled, ground_truth filled in)
  - redteam_set.csv (synthetic, pre-labelled)

Writes:
  - test_set.csv (id, content, ground_truth, source)

Rows in candidates_to_label.csv with empty ground_truth are skipped.
"""

import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CANDIDATES = HERE / "candidates_to_label.csv"
REDTEAM = HERE / "redteam_set.csv"
OUT = HERE / "test_set.csv"


def load(path: Path, source: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            gt = r.get("ground_truth", "").strip()
            if gt not in ("0", "1"):
                continue
            rows.append({
                "content": r["content"],
                "ground_truth": gt,
                "source": source,
                "subgroup": r.get("tier") or r.get("category") or "",
            })
    return rows


def main():
    cands = load(CANDIDATES, "bsky") if CANDIDATES.exists() else []
    rt = load(REDTEAM, "redteam") if REDTEAM.exists() else []
    all_rows = cands + rt

    print(f"bluesky labelled: {len(cands)}")
    print(f"redteam:          {len(rt)}")
    print(f"total:            {len(all_rows)}")
    print(f"label balance:    {dict(Counter(r['ground_truth'] for r in all_rows))}")
    print(f"by source/label:  {dict(Counter((r['source'], r['ground_truth']) for r in all_rows))}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "content", "ground_truth", "source", "subgroup"])
        for i, r in enumerate(all_rows, 1):
            w.writerow([i, r["content"], r["ground_truth"], r["source"], r["subgroup"]])
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
