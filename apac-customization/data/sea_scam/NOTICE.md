# SEA scam/spam set — REAL data (replaces the earlier synthetic SEA set)

`sea_scam.csv` and `mlx/*.jsonl` are derived from two real datasets and are **not
vendored** here (regenerate with `data/build_sea_real.py`):

- **Thai — `tu_scam_dataset`** (`~/Downloads/tu_scam_dataset.csv.xlsx`): Thai messages
  labeled `scam` / `normal` / `suspicious` with category and signal flags. We use a
  balanced binary **scam (1) vs normal (0)** split for train/valid/test; `suspicious`
  is excluded from the binary metric. Provides both a real Thai eval and a real Thai
  fine-tune. (Placeholder domains like `example.invalid` indicate a constructed/
  benchmark corpus — real Thai text, curated labels.)
- **Tagalog/Filipino — `SPAM_SMS.csv`** (`~/Downloads/SPAM_SMS.csv`): real Philippine
  SMS spam. **Positive-only** (no benign labels), so Tagalog is scored **recall-only**
  (of real spam, how much each model catches) — no precision/FP number is claimed for
  Tagalog.

This replaces the earlier hand-authored synthetic SEA moneylending set for the SEA
column of the matrix. If you publish, confirm the license/terms of each source dataset
and cite it appropriately.

NSFW/PII note: real messaging data may contain scam lures and phone-number-shaped text.
