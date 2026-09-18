#!/bin/bash
# Re-run Shieldstral (now test-only, bug fixed) on all 3 sets + finish SEA-Guard on
# Chinese + Bengali (Thai/Tagalog already done). Saves predictions for metrics.py.
set -e
PY=../.venv/bin/python
F="Fetching|it/s|it\]|Loading|weights|warn|deprecated|frames|docstring"

echo "########## Shieldstral (test-only) ##########"
$PY demo1_shieldstral/run_shieldstral.py --test data/sea_scam/sea_scam.csv --domain scam 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"
$PY demo1_shieldstral/run_shieldstral.py --test data/chifraud/chifraud_sample.csv --domain moneylending 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"
$PY demo1_shieldstral/run_shieldstral.py --test data/bengali/bengali_offensive.csv --domain offensive 2>&1 | grep -vE "$F" | grep -E "overall|recall|FP"

echo "########## SEA-Guard (Chinese + Bengali; Thai/PH already done) ##########"
$PY rung3_regional/run_seaguard.py --limit 300 2>&1 | grep -vE "$F" | grep -E "recall|scam|fraud|offensive|spam"

echo "########## SCORE ALL ##########"
$PY metrics.py
