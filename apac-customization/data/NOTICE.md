# Dataset notice — SYNTHETIC demo data

`apac_moneylending.csv` and `mlx/*.jsonl` are **synthetic, hand-authored
illustrative data** created by Claude (Claude Code, Opus 4.8) with Juliet Shen
for a presentation on customizing open models for APAC languages and cultural
norms. Regenerate with `python data/build_dataset.py` (deterministic seed).

- **Not scraped** from any platform; **not a validated benchmark**; no real users.
- The **non-English phrasing is best-effort and must be reviewed by a native
  speaker** before being shown or quoted publicly.
- Any precision/recall/F1 this set produces demonstrates the **customization
  mechanics** (a base model missing a region-specific harm; fine-tuning teaching
  it) — it is **not** evidence of real-world accuracy.
- Domain: advertising of **illegal / unlicensed moneylending** ("loan shark" /
  "ah long" / "5-6" / "pinjaman along"). Chosen because it is a real
  high-priority Southeast Asian Trust & Safety harm whose markers are
  language- and culture-specific, and because it cleanly separates an illegal
  advertisement from a legitimate **licensed** lender (the hard-negative class).

For a real deployment, replace this with platform data labeled by regional
trust & safety experts, with inter-annotator agreement, per the bias caveats in
the parent repo's `README.md`.
