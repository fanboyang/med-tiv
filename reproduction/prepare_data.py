#!/usr/bin/env python3
"""准备 Med-TIV 的 verl 训练池与独立评测题集。

已于 2026-11-03 用 HF dataset viewer 核对下列 schema。MedPRM 的
solutions[].orm_label 是 outcome 标签，不能解释成人工逐步错误监督。
仅使用原始 solution；参考答案、PRM 标签和 related_docs 不进入 prompt。
依赖 datasets/pyarrow；纯转换与离线单元测试只依赖标准库。
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import unicodedata

MEDPRM = {
    'dataset': 'dmis-lab/llama-3.1-medprm-reward-training-set',
    'revision': 'da371b3d91234be356b44c9cea56f1e39ead112e',
    'config': 'default', 'split': 'train',
}
MMLU_SUBJECTS = ('anatomy', 'clinical_knowledge', 'college_biology',
                 'college_medicine', 'medical_genetics', 'professional_medicine')
EVALUATION_SOURCES = {
    'medqa': {'dataset': 'GBaker/MedQA-USMLE-4-options',
              'revision': '0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18',
              'config': 'default', 'split': 'test'},
    # 公开 test 不提供可用金标；采用论文常用的有标注 validation。
    'medmcqa': {'dataset': 'openlifescienceai/medmcqa',
                'revision': '91c6572c454088bf71b679ad90aa8dffcd0d5868',
                'config': 'default', 'split': 'validation'},
    'mmlu_med': {'dataset': 'cais/mmlu',
                 'revision': 'c30699e8356da336a370243923dbaf21066bb9fe',
                 'configs': MMLU_SUBJECTS, 'split': 'test'},
    'medxpertqa': {'dataset': 'TsinghuaC3I/MedXpertQA',
                  'revision': '7e7c465a68eb2b866926bfa59c8c9d17a8daba65',
                  'config': 'Text', 'split': 'test'},
}

# 论文 Table 4；仅合并 PDF 换行，不增加金标或参考答案。
PROMPT_TEMPLATE = '''You are a reasoning validator for medical problems. Your task is to think step by step and evaluate whether the given reasoning trace of a medical problem contains errors. First, you must always perform a step-by-step analysis to examine the entire reasoning process. Then, based on your analysis, you will make a definitive judgment.
- Use 1 if the reasoning trace is free of errors.
- Use 0 if the reasoning trace contains one or more errors.
Output Instruction:
You must conduct your step-by-step analysis inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.
Medical Problem:
{problem}

Reasoning Trace:
{trace}'''


def question_key(question):
    """按 NFKC、大小写及空白归一化题干分组；不计算文件哈希。"""
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} 必须为非空字符串')
    return value.strip()


def medprm_records(rows, excluded_questions=(), source_split='train'):
    if source_split != 'train':
        raise ValueError('训练池只允许 MedPRM train split')
    excluded = {question_key(q) for q in excluded_questions}
    for row_idx, row in enumerate(rows):
        question = text(row.get('question'), 'question')
        key = question_key(question)
        if key in excluded:
            continue
        options = row.get('options')
        if not isinstance(options, list) or not 2 <= len(options) <= 26:
            raise ValueError(f'row {row_idx}: options 必须是选项列表')
        # 固定revision中存在上游解析损坏的空字符串；保留原字母与原文本，
        # 不删除、不补全、不用参考答案修复。此规则仅用于原始MedPRM训练数据。
        for option_idx, option in enumerate(options):
            if not isinstance(option, str):
                raise ValueError(f'row {row_idx}, option {option_idx}: 必须为字符串')
        empty_option_indices = [i for i, option in enumerate(options) if not option.strip()]
        rendered = '\n'.join(f'({chr(65+i)}) {option}' for i, option in enumerate(options))
        solutions = row.get('solutions')
        if not isinstance(solutions, list):
            raise ValueError(f'row {row_idx}: solutions 必须是列表')
        # HF 实际数据存在 solutions=[]：该问题没有可供训练的 trace。
        for solution_idx, solution in enumerate(solutions):
            label = solution.get('orm_label')
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f'row {row_idx}, solution {solution_idx}: 缺少二元 orm_label')
            trace = text(solution.get('solution'), 'solutions[].solution')
            yield {
                'data_source': 'med_tiv',
                'prompt': [{'role': 'user', 'content': PROMPT_TEMPLATE.format(
                    problem=f'Question: {question}\nOptions:\n{rendered}', trace=trace)}],
                'reward_model': {'style': 'rule', 'ground_truth': str(label)},
                'extra_info': {
                    'id': f'medprm:{row_idx}:{solution_idx}',
                    'question_id': str(row.get('question_id', row_idx)),
                    'question_key': key, 'solution_index': solution_idx,
                    'original_data_source': str(row.get('data_source', '')),
                    'source_split': 'train', 'split': '',
                    'label_source': 'solutions.orm_label',
                    'empty_option_indices': empty_option_indices,
                    'source_option_count': len(options),
                    'unusual_option_count': len(options) not in (3, 4, 5),
                },
            }


def split_by_question(records, validation_fraction=.05, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction 必须在 (0,1)')
    keys = sorted({r['extra_info']['question_key'] for r in records})
    if len(keys) < 2:
        raise ValueError('至少需要两道独立题目以隔离内部验证集')
    random.Random(seed).shuffle(keys)
    n = max(1, min(len(keys)-1, round(len(keys)*validation_fraction)))
    validation_keys = set(keys[:n])
    train, validation = [], []
    for record in records:
        is_validation = record['extra_info']['question_key'] in validation_keys
        copied = {**record, 'extra_info': {**record['extra_info'],
                  'split': 'validation' if is_validation else 'train'}}
        (validation if is_validation else train).append(copied)
    return train, validation


def balanced_sample(records, budget, seed):
    if budget <= 0 or budget % 2:
        raise ValueError('平衡预算必须为正偶数')
    groups = [[r for r in records if r['reward_model']['ground_truth'] == str(label)]
              for label in (0, 1)]
    if any(len(g) < budget // 2 for g in groups):
        raise ValueError(f'无法无放回抽取平衡预算 {budget}，各标签可用数 {[len(g) for g in groups]}')
    rng = random.Random(seed)
    result = rng.sample(groups[0], budget//2) + rng.sample(groups[1], budget//2)
    rng.shuffle(result)
    return result


def normalize_evaluation(name, row, index, split, subject=''):
    question = text(row.get('question'), 'question')
    if name in ('medqa', 'medxpertqa'):
        options = row.get('options')
        gold = row.get('answer_idx' if name == 'medqa' else 'label')
        if name == 'medxpertqa':
            question = question.split('\nAnswer Choices:', 1)[0].strip()
    elif name in ('medmcqa', 'mmlu_med'):
        choices = ([row.get('op'+c) for c in 'abcd'] if name == 'medmcqa'
                   else row.get('choices'))
        if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
            raise ValueError(f'{name}: 无效选项列表')
        options = {chr(65+i): value for i, value in enumerate(choices)}
        answer = row.get('cop' if name == 'medmcqa' else 'answer')
        if type(answer) is not int or not 0 <= answer < len(options):
            raise ValueError(f'{name}: 金标必须为有效的零基索引')
        gold = chr(65+answer)
    else:
        raise ValueError(f'未知评测来源 {name}')
    if not isinstance(options, dict) or not options or len(options) > 26:
        raise ValueError(f'{name}: options 必须为非空字典')
    options = {k: text(options[k], 'option') for k in sorted(options)}
    if list(options) != [chr(65+i) for i in range(len(options))] or gold not in options:
        raise ValueError(f'{name}: 金标或选项字母无效')
    return {'id': f'{name}:{subject}:{row.get("id", index)}', 'question': question,
            'options': options, 'gold': gold, 'dataset': name, 'split': split,
            'subject': subject}


def load_source(source, config, cache_dir=None, offline=False, local_path=None):
    """HF 默认使用本地缓存；offline 严禁网络。支持 save_to_disk 和本地 JSONL/parquet。

    local-sources JSON 的键为 medprm、medqa、medmcqa、medxpertqa，及
    mmlu_med/anatomy 等六个键；每项值是本地路径，不推断或更换 split。
    """
    if offline:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    from datasets import DownloadConfig, load_dataset, load_from_disk
    if local_path:
        path = Path(local_path).expanduser().resolve(strict=True)
        if path.is_dir() and ((path/'dataset_info.json').exists() or (path/'dataset_dict.json').exists()):
            loaded = load_from_disk(str(path))
            if hasattr(loaded, 'keys'):
                return loaded[source['split']]
            return loaded
        if path.suffix in ('.jsonl', '.json', '.parquet'):
            kind = 'parquet' if path.suffix == '.parquet' else 'json'
            return load_dataset(kind, data_files={source['split']: str(path)},
                                split=source['split'], cache_dir=cache_dir)
        # HF 的本地 snapshot 目录：由 datasets 按数据仓库本身配置读取。
        return load_dataset(str(path), name=config, split=source['split'], cache_dir=cache_dir,
                            download_config=DownloadConfig(local_files_only=offline))
    return load_dataset(source['dataset'], name=config, revision=source['revision'],
                        split=source['split'], cache_dir=cache_dir,
                        download_config=DownloadConfig(local_files_only=offline))


def data_quality_summary(records):
    quality = {}
    for issue, predicate in (
        ('empty_options', lambda info: bool(info['empty_option_indices'])),
        ('unusual_option_count', lambda info: info['unusual_option_count']),
    ):
        affected = [r for r in records if predicate(r['extra_info'])]
        question_ids = sorted({r['extra_info']['question_id'] for r in affected})
        quality[issue] = {'questions': len(question_ids), 'traces': len(affected),
                          'question_ids': question_ids}
    unusual_questions = {r['extra_info']['question_id']: r['extra_info']['source_option_count']
                         for r in records if r['extra_info']['unusual_option_count']}
    quality['unusual_question_option_counts'] = unusual_questions
    quality['policy'] = 'retain original strings and option letters; no reconstruction or sample removal'
    quality['scope'] = 'retained MedPRM trace pool after evaluation-question exclusion, before split'
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--local-sources', type=Path)
    parser.add_argument('--validation-fraction', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-budget', type=int, help='可选：训练split无放回平衡抽样；默认输出完整池')
    args = parser.parse_args(argv)
    local_sources = json.loads(args.local_sources.read_text()) if args.local_sources else {}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output/'eval').mkdir(exist_ok=True)
    manifest = {'schema_verified': '2026-11-03', 'seed': args.seed,
                'validation_fraction': args.validation_fraction,
                'medprm_source': MEDPRM, 'evaluation_sources': EVALUATION_SOURCES,
                'label_source': 'solutions.orm_label (outcome; not step-error annotation)',
                'empty_solutions_policy': 'valid empty list contributes zero traces; missing field fails',
                'medmcqa_protocol': 'public validation with gold labels; not unlabeled test',
                'local_sources': local_sources, 'evaluation_counts': {}}
    eval_questions = set()
    for name, source in EVALUATION_SOURCES.items():
        count = 0
        with (output/'eval'/f'{name}.jsonl').open('w') as stream:
            for config in source.get('configs', [source.get('config')]):
                local_key = f'{name}/{config}' if name == 'mmlu_med' else name
                dataset = load_source(source, config, args.cache_dir, args.offline, local_sources.get(local_key))
                for index, row in enumerate(dataset):
                    normalized = normalize_evaluation(name, row, index, source['split'],
                                                      config if name == 'mmlu_med' else '')
                    stream.write(json.dumps(normalized, ensure_ascii=False)+'\n')
                    eval_questions.add(question_key(normalized['question']))
                    count += 1
        if not count:
            raise ValueError(f'{name}: 评测集为空')
        manifest['evaluation_counts'][name] = count
    dataset = load_source(MEDPRM, MEDPRM['config'], args.cache_dir, args.offline, local_sources.get('medprm'))
    records = list(medprm_records(dataset, excluded_questions=eval_questions))
    quality = data_quality_summary(records)
    manifest['medprm_data_quality'] = quality
    if quality['empty_options']['traces'] or quality['unusual_option_count']['traces']:
        print('WARNING: MedPRM source has malformed options; preserved unchanged: '
              + json.dumps(quality, ensure_ascii=False), flush=True)
    train, validation = split_by_question(records, args.validation_fraction, args.seed)
    manifest['available_train_traces'] = len(train)
    if args.train_budget is not None:
        train = balanced_sample(train, args.train_budget, args.seed)
    from datasets import Dataset
    for split, rows in [('train', train), ('validation', validation)]:
        Dataset.from_list(rows).to_parquet(str(output/f'{split}.parquet'))
        manifest[split] = {'traces': len(rows),
                           'questions': len({r['extra_info']['question_key'] for r in rows}),
                           'label_counts': dict(Counter(r['reward_model']['ground_truth'] for r in rows))}
    manifest['excluded_eval_question_keys'] = len(eval_questions)
    manifest['training_split'] = 'MedPRM train only; normalized question disjoint internal validation'
    manifest['decontamination'] = 'NFKC/case/whitespace normalized exact question; paraphrases not detected'
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
