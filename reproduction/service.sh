#!/usr/bin/env bash
set -euo pipefail
root=${MEDTIV_ROOT:-/data/fby/med-tiv}
mode=${1:?retriever, tool, verifier, or generator required}
model=${2:-$root/models/Qwen2.5-7B-Instruct}
cd "$root/code"
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export no_proxy="localhost,127.0.0.1,0.0.0.0"
source "$root/env/bin/activate"
case "$mode" in
  retriever)
    export CUDA_VISIBLE_DEVICES=4
    exec "$root/retriever-env/bin/python" -u verl_tool/servers/tools/utils/retrieval_server.py \
      --index_path "$root/data/retriever_index/medcpt_Flat.index" \
      --corpus_path "$root/data/retriever_index/medical_combined.jsonl" \
      --retriever_name medcpt --retriever_model "$root/models/MedCPT-Query-Encoder" \
      --retrieval_pooling_method cls --topk 3 --faiss_gpu --host 127.0.0.1 --port 18501
    ;;
  tool)
    export CUDA_VISIBLE_DEVICES=''
    export RETRIEVER_URL=http://127.0.0.1:18501 RETRIEVER_CACHE_SIZE=100000
    exec python -u -m verl_tool.servers.serve --host 127.0.0.1 --port 18502 \
      --tool_type search_retrieval --workers_per_tool 8
    ;;
  verifier)
    export CUDA_VISIBLE_DEVICES=5
    exec python -u eval_service/app.py --model "$model" --host 127.0.0.1 --port 18500 \
      --workers 1 --num_models 1 --tensor_parallel_size 1 --gpu_memory_utilization 0.6 \
      --max_model_len 16384 --tool_server_url http://127.0.0.1:18502/get_observation \
      --max_turns 4 --max_obs_length 2048 --action_stop_tokens '</search>,</answer>'
    ;;
  generator)
    export CUDA_VISIBLE_DEVICES=0
    exec vllm serve "$root/models/Qwen2.5-7B-Instruct" --host 127.0.0.1 --port 18503 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.6 --max-model-len 16384 \
      --disable-log-requests
    ;;
  *) echo "unknown service: $mode" >&2; exit 2 ;;
esac
