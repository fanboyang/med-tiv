#!/usr/bin/env bash
# Only network/rate-limit failures are retried; corpus revisions stay fixed.
set -uo pipefail
root=${MEDTIV_ROOT:-/data/fby/med-tiv}
cd "$root"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$root/cache/huggingface" HF_HUB_DISABLE_XET=1
run=$(date -u +%Y%m%dT%H%M%SZ)
for attempt in 1 2 3 4 5 6; do
  log="$root/logs/corpus-${run}-attempt${attempt}.log"
  bootstrap-env/bin/python code/reproduction/download_corpus.py \
    --output-dir data/retriever_index --cache-dir cache/datasets --num-proc 1 >"$log" 2>&1
  status=$?
  printf '%s\n' "$status" >"${log%.log}.exit"
  if [[ "$status" == 0 ]]; then
    echo "CORPUS_COMPLETE $log"
    exit 0
  fi
  if ! grep -Eq '429 Client Error|ConnectionError|Connection reset|ReadTimeout|ConnectTimeout|Temporary failure in name resolution|502 Server Error|503 Server Error|504 Server Error' "$log"; then
    echo "TERMINAL_FAILURE $log"
    exit "$status"
  fi
  echo "NETWORK_FAILURE attempt=$attempt log=$log"
  [[ "$attempt" == 6 ]] && exit "$status"
  sleep 330
done
