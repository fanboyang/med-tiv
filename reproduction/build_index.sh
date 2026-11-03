#!/usr/bin/env bash
set -euo pipefail
root=${MEDTIV_ROOT:-/data/fby/med-tiv}
cd "$root/code"
source "$root/env/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
corpus="$root/data/retriever_index"
export MEDTIV_CORPUS="$corpus"
python - <<'PY'
import json, os, subprocess
from pathlib import Path
p=Path(os.environ['MEDTIV_CORPUS'])
m=json.loads((p/'medical_combined.manifest.json').read_text())
assert set(m['counts']) == {'pubmed','textbooks'} and all(n>0 for n in m['counts'].values()), m
assert m['total_documents'] == sum(m['counts'].values())
assert not (p/'medcpt_Flat.index').exists(), 'final index already exists; inspect before rebuilding'
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu, memory=map(int,line.split(','))
    if gpu in range(4) and memory > 1024:
        raise RuntimeError(f'GPU {gpu} occupied: {memory} MiB; refusing to contend')
PY
python -u build_index.py \
  --corpus_path "$corpus/medical_combined.jsonl" \
  --index_output "$corpus/medcpt_Flat.index.partial" \
  --model "$root/models/MedCPT-Article-Encoder" \
  --model_name_tag medcpt --pooling cls --batch_size 512 --use_fp16 --gpus 0,1,2,3
python - <<'PY'
import faiss, json, os
from pathlib import Path
p=Path(os.environ['MEDTIV_CORPUS'])
m=json.loads((p/'medical_combined.manifest.json').read_text())
source=p/'medcpt_Flat.index.partial'
index=faiss.read_index(str(source))
assert index.ntotal == m['total_documents'], (index.ntotal,m['total_documents'])
assert index.d == 768, index.d
meta=json.loads(Path(str(source)+'.meta.json').read_text())
assert meta['total_indexed'] == index.ntotal
meta['index_path']=str(p/'medcpt_Flat.index')
meta['verified_corpus_counts']=m['counts']
Path(str(source)+'.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
os.replace(source,p/'medcpt_Flat.index')
os.replace(str(source)+'.meta.json',p/'medcpt_Flat.index.meta.json')
print('INDEX_VERIFIED',index.ntotal,index.d)
PY
