#!/usr/bin/env bash
# Off-topic (null) sweep: pair each aligned baseline with the violent-extremism
# off-topic policy, on both domains, for all three models. Under the off-topic
# policy the correct answer is 0 on every row — any flag is topic leakage.
#
# Read results with offtopic.py, NOT the summary_*.csv (ground_truth encodes the
# wrong harm for the off-topic column, so its F1/accuracy is meaningless).
#
# cope_a / cope_b hit their Modal endpoints (must be deployed & warm).
# shieldstral runs locally (SUPPORTS_CONCURRENCY=False -> sequential).

set -euo pipefail
cd "$(dirname "$0")"

SH_SET="test_set.csv"
SEX_SET="sexual_content_eval/test_set.csv"
OFFTOPIC="violent_extremism_offtopic"

run () {  # model  label  test_set  baseline
  local model="$1" label="$2" set="$3" base="$4"
  echo "===== $model / $label ====="
  python eval.py --model "$model" --label "$label" --test-set "$set" \
    --policies "$base" "$OFFTOPIC" --concurrency 16 \
    2>&1 | tee "results/run_${label}.log"
}

# Shieldstral (local, sequential — concurrency flag is ignored by the adapter)
run shieldstral shieldstral_sh_offtopic  "$SH_SET"  simple
run shieldstral shieldstral_sex_offtopic "$SEX_SET" sexual_content_simple

# cope-a (Gemma-2 LoRA, /v1/completions)
run cope_a cope_a_sh_offtopic  "$SH_SET"  simple
run cope_a cope_a_sex_offtopic "$SEX_SET" sexual_content_simple

# cope-b (standalone classifier, /v1/chat/completions)
run cope_b cope_b_sh_offtopic  "$SH_SET"  simple
run cope_b cope_b_sex_offtopic "$SEX_SET" sexual_content_simple

echo
echo "ALL OFF-TOPIC SWEEPS DONE"
echo "readout, e.g.:"
echo "  python offtopic.py 'results/predictions_shieldstral_sh_offtopic_*.csv' --baseline simple --offtopic $OFFTOPIC"
