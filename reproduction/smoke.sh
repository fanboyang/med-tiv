#!/usr/bin/env bash
# Fixed bounded curriculum + one update + four-example integration validation.
set -euo pipefail
root=${MEDTIV_ROOT:-/data/fby/med-tiv}
source "$root/env/bin/activate"
cd "$root/code"
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=''
export MEDTIV_SMOKE_DEVICES=0
model="$root/models/Qwen2.5-7B-Instruct"
output="$root/outputs/smoke/integration-v1"
export MEDTIV_RAY_TMPDIR="$root/tmp/smoke"
export MEDTIV_RUNTIME_DIR="$output/runtime" MEDTIV_RUN_NAME=medtiv-integration-smoke-v1
python - "$root" "$output" <<'PY'
import json,random,subprocess,sys,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
root,out=map(Path,sys.argv[1:])
if out.exists():raise FileExistsError('smoke output already exists; inspect prior run before continuing')
if (root/'logs/full-retrieval-readiness.exit').read_text().strip()!='0':raise RuntimeError('full retrieval readiness not passed')
gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):(v[1].strip(),int(v[2])) for line in gpu_rows.splitlines() if (v:=line.split(','))}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
active={line.split(',')[0].strip() for line in apps.splitlines()}
if any(i not in gpus or gpus[i][1]>1024 or gpus[i][0] in active for i in [0]):raise RuntimeError('GPU0 are not idle')
train=pq.read_table(root/'data/prepared/train.parquet');labels=train['reward_model'].to_pylist()
indices=list(range(len(labels)));random.Random(42).shuffle(indices)
selected=[]; counts={'0':0,'1':0}
for i in indices:
    label=labels[i]['ground_truth']
    if counts[label]<8:selected.append(i);counts[label]+=1
    if len(selected)==16:break
assert counts=={'0':8,'1':8}
validation=pq.read_table(root/'data/prepared/validation.parquet')
out.mkdir(parents=True)
pq.write_table(train.take(selected),out/'pool.parquet')
pq.write_table(validation.slice(0,4),out/'validation.parquet')
(out/'manifest.json').write_text(json.dumps({'scope':'integration_smoke_not_benchmark','pool_indices':selected,'source_ids':[row['id'] for row in train.take(selected)['extra_info'].to_pylist()],'seed':42,'curriculum_n':8,'curriculum_budget':4,'optimizer_steps':1,'validation_rows':4},indent=2))
PY
python -m reproduction.curriculum --pool "$output/pool.parquet" --model "$model" \
  --output-dir "$output/curriculum" --budget 4 --workers 1 --seed 42
python - <<'GPUCHECK'
import subprocess
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):(v[1].strip(),int(v[2])) for line in rows.splitlines() if (v:=line.split(','))}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
active={line.split(',')[0].strip() for line in apps.splitlines()}
if any(i not in gpus or gpus[i][1]>1024 or gpus[i][0] in active for i in [0]):raise RuntimeError('GPU0 became occupied during curriculum')
GPUCHECK
bash reproduction/train.sh 1 "$model" "$output/curriculum/train.parquet" \
  data.val_files="$output/validation.parquet" \
  data.train_batch_size=4 data.val_batch_size=4 data.dataloader_num_workers=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false}' \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  trainer.total_training_steps=1 trainer.total_epochs=1 \
  trainer.test_freq=1 trainer.save_freq=-1 trainer.val_before_train=False \
  trainer.resume_mode=disable trainer.experiment_name=medtiv-integration-smoke-v1 \
  trainer.default_local_dir="$output/checkpoints" trainer.logger=['console']
