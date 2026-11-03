"""Full-denominator Med-TIV evaluation; successful and terminal-failure entries are resumed.

Generator max_tokens=8192 is a reproduction choice (paper does not specify it).
Verifier max_tokens=1024 is the author's per-turn default; server manages tool turns.
The author eval_service /completions expects an already chat-templated prompt.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import urllib.request
from urllib.error import HTTPError, URLError
from reproduction.reward import strip_observations


def option_map(row):
    opts = row['options']
    if isinstance(opts, list):
        opts = {chr(65+i): value for i, value in enumerate(opts)}
    if not isinstance(opts, dict) or not opts:
        raise ValueError('options must be a nonempty dict or list')
    return {str(k): str(v) for k,v in opts.items()}


def generator_messages(row):
    options = option_map(row)
    content = row['question'] + '\n' + '\n'.join(f'{k}. {v}' for k,v in options.items())
    content += '\nReason carefully. Put only the final option letter inside <answer>...</answer>.'
    return [{'role':'user', 'content':content}]


def parse_answer(text, allowed):
    matches = re.findall(r'<answer>\s*([^<>]+?)\s*</answer>', text)
    if len(matches) != 1:
        return None
    return matches[0] if matches[0] in allowed else None


def model_generated_text(raw):
    """Require the service's model-only field; full text may contain truncated tools."""
    text = raw['choices'][0]['generated_text']
    if not isinstance(text, str):
        raise ValueError('service generated_text must be a string')
    return text


def parse_verdict(text):
    text = strip_observations(text)
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not matches or matches[-1].strip() not in ('0','1'):
        return None
    return int(matches[-1].strip())


def verifier_messages(row, trace):
    question = row['question'] + '\n' + '\n'.join(f'{k}. {v}' for k,v in option_map(row).items())
    return [{'role':'user', 'content':(
        'You are a reasoning validator for medical problems. Your task is to think step by step '
        'and evaluate whether the given reasoning trace of a medical problem contains errors.\n'
        'First, you must always perform a step-by-step analysis to examine the entire reasoning process. '
        'Then, based on your analysis, you will make a definitive judgment.\n'
        '- Use 1 if the reasoning trace is free of errors.\n'
        '- Use 0 if the reasoning trace contains one or more errors.\n\n'
        'You must conduct your step-by-step analysis inside <think> and </think> first every time '
        'you get new information. After reasoning, if you find you lack some knowledge, you can '
        'call a search engine by <search> query </search> and it will return the top searched '
        'results between <information> and </information>. You can search as many times as you want. '
        'If you find no further external knowledge needed, you can directly provide the answer '
        'inside <answer> and </answer>, without detailed illustrations.\n\n'
        f'Medical Problem:\n{question}\n\nReasoning Trace:\n{trace}') }]


def hard_vote(candidates):
    if not candidates or any(a is None or label not in (0,1) for a,label in candidates):
        return None, False
    approved = [a for a,label in candidates if label == 1]
    rejected_all = not approved
    pool = approved if approved else [a for a,_ in candidates]
    # Author implementation: ties are resolved by first occurrence, fixed candidate order.
    return Counter(pool).most_common(1)[0][0], rejected_all


def summarize(rows):
    total = len(rows)
    correct = sum(r['prediction'] is not None and r['prediction'] == r['gold'] for r in rows)
    return {'total':total, 'correct':correct, 'accuracy':correct/total if total else None,
            'missing':sum(r['prediction'] is None for r in rows),
            'all_rejected':sum(r['all_rejected'] for r in rows),
            'missing_policy':'any_missing_candidate_makes_question_missing'}


class ResponseDecodeError(ValueError):
    def __init__(self, raw_response):
        super().__init__('invalid JSON response')
        self.raw_response = raw_response


def post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode('utf-8', errors='replace')
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResponseDecodeError(raw) from exc


class Journal:
    def __init__(self, path, config):
        self.path = path
        self.lock = threading.Lock()
        self.completed = {}
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    record = json.loads(line)  # Damaged journals fail explicitly.
                    if record['kind'] == 'config':
                        if record['config'] != config:
                            raise ValueError('journal configuration differs; use a new output directory')
                    elif record.get('status') in ('ok','terminal_failure'):
                        self.completed[tuple(record['key'])] = record
                    elif record.get('status') == 'error':
                        raise ValueError('legacy ambiguous error record; use a new output directory')
        else:
            self.append({'kind':'config','config':config})

    def append(self, record):
        with self.lock:
            with self.path.open('a') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                handle.flush()
            if record.get('status') in ('ok','terminal_failure'):
                self.completed[tuple(record['key'])] = record

    def call(self, key, url, body, extract, timeout):
        key = tuple(key)
        if key in self.completed:
            return self.completed[key].get('value')
        record = {'kind':'request','key':list(key),'request':body}
        try:
            raw = post(url,body,timeout)
            record['raw_response'] = raw
            value = extract(raw)
            if value is None:
                raise ValueError('invalid or missing structured answer')
            record.update(status='ok',value=value)
        except HTTPError as exc:
            # Only server errors are retryable; 4xx requests require explicit repair.
            record.update(status='retryable_failure' if 500 <= exc.code < 600 else 'terminal_failure',
                          error_type=type(exc).__name__, http_status=exc.code,
                          raw_response=exc.read().decode('utf-8',errors='replace'))
            value = None
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            record.update(status='retryable_failure',error_type=type(exc).__name__)
            value = None
        except Exception as exc:
            record.update(status='terminal_failure',error_type=type(exc).__name__)
            if isinstance(exc, ResponseDecodeError):
                record['raw_response'] = exc.raw_response
            value = None
        self.append(record)
        return value


def run(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.verifier_tokenizer, local_files_only=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    inputs = {}
    for spec in args.dataset:
        name, path = spec.split('=',1)
        if name in inputs:
            raise ValueError('duplicate dataset name')
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or len({str(r['id']) for r in rows}) != len(rows):
            raise ValueError('empty dataset or duplicate IDs')
        for row in rows:
            if row['gold'] not in option_map(row):
                raise ValueError('gold is not an option key')
        inputs[name] = rows
    config = {k:v for k,v in vars(args).items() if k not in ['workers','timeout','output_dir']}
    config['protocol_version'] = 4
    config['verifier_server_response_budget'] = 8192
    config['verifier_server_budget_source'] = 'original model IDs + retained observation IDs'
    config['verifier_text_source'] = 'choices[0].generated_text'
    config['missing_policy'] = 'any_missing_candidate_makes_question_missing'
    config['input_rows'] = inputs  # Exact resume identity, no data hash.
    journal = Journal(output/'requests.jsonl',config)

    def question_job(job):
        name,row,seed = job
        results = []
        for index in range(args.candidates):
            key = [name,str(row['id']),seed,index]
            sample_seed = seed + index*1000003
            body = {'model':args.generator_model,'messages':generator_messages(row),
                    'temperature':args.generator_temperature,'top_p':args.generator_top_p,
                    'max_tokens':args.generator_max_tokens,'seed':sample_seed,'n':1}
            def extract_generation(raw):
                text = raw['choices'][0]['message']['content']
                answer = parse_answer(text,option_map(row))
                return {'trace':text,'answer':answer} if answer is not None else None
            generated = journal.call(key+['generate'],args.generator_url.rstrip('/')+'/chat/completions',
                                     body,extract_generation,args.timeout)
            if generated is None:
                results.append((None,None))
                continue
            prompt = tokenizer.apply_chat_template(verifier_messages(row,generated['trace']),
                                                   tokenize=False,add_generation_prompt=True)
            verify_body = {'model':args.verifier_model,'prompt':prompt,'n':1,
                           'temperature':args.verifier_temperature,'top_p':args.verifier_top_p,'max_tokens':args.verifier_max_tokens,
                           'seed':sample_seed}
            verdict = journal.call(key+['verify'],args.verifier_url.rstrip('/')+'/completions',
                                   verify_body,lambda raw:parse_verdict(model_generated_text(raw)),args.timeout)
            results.append((generated['answer'],verdict))
        prediction,rejected_all = hard_vote(results)
        return {'dataset':name,'id':str(row['id']),'seed':seed,'prediction':prediction,
                'gold':row['gold'],'all_rejected':rejected_all,
                'missing_candidates':sum(a is None or v is None for a,v in results)}

    jobs = [(name,row,seed) for name,rows in inputs.items() for seed in args.seeds for row in rows]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(question_job,jobs))
    with (output/'predictions.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row)+'\n')
    summary = {name:{str(seed):summarize([r for r in rows if r['dataset']==name and r['seed']==seed])
                     for seed in args.seeds} for name in inputs}
    summary['complete'] = all(r['prediction'] is not None for r in rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',action='append',required=True,help='NAME=JSONL; repeat for four benchmarks')
    parser.add_argument('--generator-url',required=True,help='vLLM base URL including /v1')
    parser.add_argument('--verifier-url',required=True,help='author eval_service base URL (no /v1)')
    parser.add_argument('--generator-model',required=True)
    parser.add_argument('--verifier-model',required=True)
    parser.add_argument('--verifier-tokenizer',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--candidates',type=int,default=32)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--generator-max-tokens',type=int,default=8192)
    parser.add_argument('--verifier-max-tokens',type=int,default=1024)
    parser.add_argument('--generator-temperature',type=float,default=1.0)
    parser.add_argument('--generator-top-p',type=float,default=1.0)
    parser.add_argument('--verifier-temperature',type=float,default=0.0)
    parser.add_argument('--verifier-top-p',type=float,default=1.0)
    args=parser.parse_args()
    if args.candidates < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive candidates/workers and distinct seeds required')
    raise SystemExit(run(args))

if __name__=='__main__': main()
