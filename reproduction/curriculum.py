"""Paper curriculum: group-8 reward variance, balanced 20K, no replacement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from reproduction.evaluate import Journal
from reproduction.reward import compute_score


def informative(rewards):
    if len(rewards) != 8:
        raise ValueError('curriculum requires exactly eight samples')
    return min(rewards) != max(rewards)


def choose_balanced(rows, budget):
    if budget <= 0 or budget % 2:
        raise ValueError('budget must be positive and even')
    selected = []
    seen = set()
    counts = [0,0]
    for row in rows:
        if row['id'] in seen:
            raise ValueError('duplicate curriculum ID')
        seen.add(row['id'])
        label = int(row['label'])
        if label not in (0,1):
            raise ValueError('nonbinary label')
        if informative(row['rewards']) and counts[label] < budget//2:
            selected.append(row)
            counts[label] += 1
    if counts != [budget//2,budget//2]:
        raise ValueError(f'insufficient informative samples: {counts}, target {budget//2} each')
    return selected



def validate_pool(rows):
    """Preserve source IDs and exact prompt/label identity for resumable requests."""
    result=[]
    seen=set()
    for row in rows:
        source_id=row['extra_info']['id']
        if not isinstance(source_id,str) or not source_id or source_id in seen:
            raise ValueError('missing/nonstring/duplicate source curriculum ID')
        seen.add(source_id)
        target=row['reward_model']['ground_truth']
        if target not in ('0','1'):
            raise ValueError('ground_truth must be exact string 0 or 1')
        messages=row['prompt'].tolist() if hasattr(row['prompt'],'tolist') else row['prompt']
        if not isinstance(messages,list) or not messages:
            raise ValueError('prompt must be a nonempty message list')
        messages=[dict(message) for message in messages]
        if any(not isinstance(m.get('content'),str) or not isinstance(m.get('role'),str) for m in messages):
            raise ValueError('invalid prompt message')
        result.append({'id':source_id,'label':int(target),'messages':messages})
    if not result:
        raise ValueError('empty curriculum pool')
    return result


def check_length(tokenizer,prompt,max_prompt,max_action,max_model):
    count=len(tokenizer.encode(prompt,add_special_tokens=False))
    if count > max_prompt:
        raise ValueError(f'prompt has {count} tokens, limit {max_prompt}; no truncation allowed')
    if count+max_action > max_model:
        raise ValueError(f'prompt+action budget {count+max_action} exceeds context {max_model}')
    return count


def sample_group(row,index,tokenizer,journal,model,api_url,seed,max_tokens,
                 max_prompt_tokens,max_model_len,max_response_tokens):
    prompt=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
    check_length(tokenizer,prompt,max_prompt_tokens,max_tokens,max_model_len)
    samples=[]
    for k in range(8):
        body={'model':model,'prompt':prompt,'n':1,'temperature':1.0,'top_p':0.95,
              'max_tokens':max_tokens,'seed':seed+index*8+k}
        def extract(raw):
            text=raw['choices'][0]['text']
            if not isinstance(text,str):
                raise ValueError('service response text must be a string')
            # Matches training's total response cap, including tool observations.
            response_tokens=len(tokenizer.encode(text,add_special_tokens=False))
            if response_tokens > max_response_tokens:
                raise ValueError('service response exceeds training response budget')
            if len(tokenizer.encode(prompt,add_special_tokens=False))+response_tokens > max_model_len:
                raise ValueError('service response exceeds model context budget')
            return {'text':text,'reward':compute_score(text,str(row['label'])),
                    'response_tokens':response_tokens}
        value=journal.call([row['id'],k],api_url.rstrip('/')+'/completions',body,extract,900)
        if value is None:
            # Valid-HTTP model failures are terminal in Journal. Transport/5xx can be
            # resumed, but neither kind is silently replaced with another sample.
            raise RuntimeError(f'curriculum request missing: source={row["id"]}, sample={k}; inspect requests.jsonl')
        samples.append(value)
    return {'id':row['id'],'label':row['label'],'status':'ok',
            'rewards':[sample['reward'] for sample in samples],'samples':samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--api-url',default='http://127.0.0.1:18500')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--budget',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--max-prompt-tokens',type=int,default=8192)
    parser.add_argument('--max-model-len',type=int,default=16384)
    parser.add_argument('--max-response-tokens',type=int,default=8192)
    args=parser.parse_args()
    if args.budget <= 0 or args.budget % 2 or args.workers < 1:
        parser.error('positive even budget and positive workers required')
    if min(args.max_tokens,args.max_prompt_tokens,args.max_model_len,args.max_response_tokens) < 1:
        parser.error('all token budgets must be positive')
    import pandas as pd
    from transformers import AutoTokenizer
    frame=pd.read_parquet(args.pool)
    pool=validate_pool(frame.to_dict(orient='records'))
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ('workers','output_dir')}
    config['protocol_version']=2
    config['pool_rows']=pool  # Exact input identity, no hashes; reordering also fails.
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('curriculum resume config mismatch')
    config_path.write_text(json.dumps(config,indent=2))
    journal=Journal(output/'requests.jsonl',config)
    source_indices={row['id']:index for index,row in enumerate(pool)}
    order=list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    selected=[]
    counts=[0,0]

    def sample(index):
        return sample_group(pool[index],index,tokenizer,journal,args.model,args.api_url,args.seed,
                            args.max_tokens,args.max_prompt_tokens,args.max_model_len,args.max_response_tokens)

    # Worker completion order never changes the curriculum selection order. Each
    # successful request is already durably journalled before any batch can fail.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0,len(order),args.workers):
            batch=order[start:start+args.workers]
            for record in executor.map(sample,batch):
                label=record['label']
                if informative(record['rewards']) and counts[label] < args.budget//2:
                    selected.append(record)
                    counts[label]+=1
            (output/'progress.json').write_text(json.dumps({
                'processed':min(start+len(batch),len(order)), 'selected_by_label':counts,
                'budget':args.budget},indent=2))
            if counts == [args.budget//2,args.budget//2]:
                break
    chosen=choose_balanced(selected,args.budget)
    indices=[source_indices[row['id']] for row in chosen]
    frame.iloc[indices].to_parquet(output/'train.parquet',index=False)
    (output/'summary.json').write_text(json.dumps({'selected':len(chosen),'by_label':counts,
        'source_rows':len(frame),'seed':args.seed,'group_size':8,
        'selected_ids':[row['id'] for row in chosen]},indent=2))

if __name__=='__main__': main()
