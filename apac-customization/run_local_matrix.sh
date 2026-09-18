#!/bin/bash
# Re-run the full LOCAL matrix on the enlarged test sets, saving per-row predictions
# for metrics.py (P/R/F1/FPR/MCC + bootstrap CIs) and latency.
# Qwen base + per-language LoRA adapters × {Thai/PH, Chinese, Bengali}, plus Shieldstral
# and SEA-Guard. Run from the apac-customization/ dir.
set -e
PY=../.venv/bin/python
F="Fetching|it/s|it\]|Loading|weights|warn|deprecated|frames|docstring"

echo "########## Qwen3-1.7B base + per-language LoRA ##########"
# Thai/PH scam
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --test data/sea_scam/sea_scam.csv --policy scam_minimal --label sea_base 2>&1 | grep -vE "$F" | tail -1
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --adapter rung2_lora/adapters_thai --test data/sea_scam/sea_scam.csv --policy scam_minimal --label sea_tuned 2>&1 | grep -vE "$F" | tail -1
# Chinese fraud
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --test data/chifraud/chifraud_sample.csv --policy minimal --label zh_base 2>&1 | grep -vE "$F" | tail -1
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --adapter rung2_lora/adapters_chifraud --test data/chifraud/chifraud_sample.csv --policy minimal --label zh_tuned 2>&1 | grep -vE "$F" | tail -1
# Bengali offensive
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --test data/bengali/bengali_offensive.csv --policy offensive_minimal --label bn_base 2>&1 | grep -vE "$F" | tail -1
$PY run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --adapter rung2_lora/adapters_bengali --test data/bengali/bengali_offensive.csv --policy offensive_minimal --label bn_tuned 2>&1 | grep -vE "$F" | tail -1

echo "########## Shieldstral ##########"
$PY demo1_shieldstral/run_shieldstral.py --test data/sea_scam/sea_scam.csv --domain scam 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"
$PY demo1_shieldstral/run_shieldstral.py --test data/chifraud/chifraud_sample.csv --domain moneylending 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"
$PY demo1_shieldstral/run_shieldstral.py --test data/bengali/bengali_offensive.csv --domain offensive 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"

echo "########## SEA-Guard ##########"
$PY rung3_regional/run_seaguard.py --limit 300 2>&1 | grep -vE "$F" | grep -E "recall|scam|fraud|offensive|spam"

echo "########## SCORE ##########"
$PY metrics.py
