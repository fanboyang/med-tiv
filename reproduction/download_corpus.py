#!/usr/bin/env python3
"""严格准备 MedRAG textbooks + pubmed；任一源失败即非零退出。

复用作者 download_medrag.py 的 id/text/source 和 title 拼接规则。
每源只有完成遍历且行数等于 HF Dataset 长度，才将 .partial 原子发布为
.jsonl 并写 completed manifest。失败重试复用同 revision 的 HF 下载缓存；
源级转换从头重做，已完成源直接复用。无抽样、无静默跳源、无哈希复核。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

SOURCES = {
    'textbooks': {'dataset': 'MedRAG/textbooks', 'revision': '9c72838920a1323ffa867467d3f7aa7b36b0f994'},
    'pubmed': {'dataset': 'MedRAG/pubmed', 'revision': '33da3593d5756bc04c8909f170003c0b14197957'},
}
FORMAT_VERSION = 1
# 固定revision的完整chunk清单，经dataset-info siblings实际核验。
EXPECTED_CHUNKS = {
    'pubmed': tuple(f'chunk/pubmed23n{i:04d}.jsonl' for i in range(1, 1167)),
    'textbooks': tuple('chunk/'+name+'.jsonl' for name in (
        'Anatomy_Gray', 'Biochemistry_Lippincott', 'Cell_Biology_Alberts',
        'First_Aid_Step1', 'First_Aid_Step2', 'Gynecology_Novak', 'Histology_Ross',
        'Immunology_Janeway', 'InternalMed_Harrison', 'Neurology_Adams',
        'Obstentrics_Williams', 'Pathology_Robbins', 'Pathoma_Husain',
        'Pediatrics_Nelson', 'Pharmacology_Katzung', 'Physiology_Levy',
        'Psichiatry_DSM-5', 'Surgery_Schwartz')),
}


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def convert_document(item, source, index):
    if source not in SOURCES:
        raise ValueError(f'未知语料源 {source}')
    # content/text 是作者支持的两个显式schema；content存在但无效时不改取其他字段。
    content = item['content'] if 'content' in item else item.get('text')
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f'{source} row {index}: content/text 必须为非空字符串')
    title = item.get('title', '')
    if title is None:
        title = ''
    if not isinstance(title, str):
        raise ValueError(f'{source} row {index}: title 必须为字符串或null')
    doc_id = item.get('id', f'{source}_{index}')
    if not isinstance(doc_id, (str, int)) or isinstance(doc_id, bool) or str(doc_id) == '':
        raise ValueError(f'{source} row {index}: id 无效')
    return {'id': doc_id, 'text': f'{title}. {content}' if title and title not in content else content,
            'source': source}


def hf_local_files(source, num_proc=8, offline=False):
    """单次仓库元数据获取完整清单；不访问会分页的HF tree API。"""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.constants import ENDPOINT
    if num_proc <= 0:
        raise ValueError('num_proc 必须为正数')
    specification = SOURCES[source]
    endpoint = os.environ.get('HF_ENDPOINT', ENDPOINT)
    info = HfApi(endpoint=endpoint).dataset_info(specification['dataset'], revision=specification['revision'])
    names = [item.rfilename for item in info.siblings if item.rfilename.startswith('chunk/')]
    expected = EXPECTED_CHUNKS[source]
    if (info.sha != specification['revision'] or len(names) != len(expected)
            or set(names) != set(expected)):
        raise ValueError(f'{source}: 固定revision完整文件清单不符；'
                         f'expected={len(expected)}, actual={len(names)}, revision={info.sha}')

    def obtain(filename):
        # 不传cache_dir：Hub缓存沿用当前HF_HOME，与已有54GB缓存相同。
        cached = try_to_load_from_cache(specification['dataset'], filename=filename,
                                        revision=specification['revision'], repo_type='dataset')
        if isinstance(cached, str):
            path = cached
        else:
            path = hf_hub_download(repo_id=specification['dataset'], filename=filename,
                                   revision=specification['revision'], repo_type='dataset',
                                   endpoint=endpoint, local_files_only=offline)
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            raise ValueError(f'{source}: 分片未完整落盘 {filename}')
        return str(path)

    executor = ThreadPoolExecutor(max_workers=num_proc)
    futures = {executor.submit(obtain, name): name for name in sorted(names)}
    paths = {}
    try:
        for future in as_completed(futures):
            paths[futures[future]] = future.result()
    finally:
        # 任一分片失败，取消尚未开始的下载；异常继续向上抛出，不能交付部分清单。
        executor.shutdown(wait=True, cancel_futures=True)
    if len(paths) != len(expected):
        raise ValueError(f'{source}: 本地分片不完整 {len(paths)}/{len(expected)}')
    print(f'{source}: 固定revision全部 {len(paths)} 个分片已在HF缓存就绪', flush=True)
    return [paths[name] for name in sorted(names)]


def load_hf_source(source, cache_dir=None, num_proc=8, offline=False):
    if offline:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    paths = hf_local_files(source, num_proc=num_proc, offline=offline)
    from datasets import load_dataset
    # 只给JSON builder显式本地路径，彻底避免HF仓库树枚举和分页Link。
    return load_dataset('json', data_files={'train': paths}, split='train',
                        cache_dir=cache_dir, num_proc=num_proc)


def quarantine(path):
    """保留作者旧产物，但不让无可信manifest的文件继续占用成功文件名。"""
    if path.exists():
        target = path.with_name(path.name+'.unverified')
        if target.exists():
            target = path.with_name(path.name+f'.unverified.{time.time_ns()}')
        path.replace(target)
        print(f'未验证的旧产物保留为 {target}', flush=True)


def prepare_source(output_dir, source, cache_dir=None, num_proc=8, offline=False, loader=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root/f'{source}.jsonl'
    manifest_path = root/f'{source}.manifest.json'
    expected_source = SOURCES[source]
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (state.get('status') != 'complete' or state.get('format_version') != FORMAT_VERSION
                or state.get('source') != source or state.get('dataset') != expected_source['dataset']
                or state.get('revision') != expected_source['revision']):
            raise ValueError(f'{manifest_path}: 已完成源的版本或来源不匹配，请使用独立输出目录')
        if (not final.is_file() or final.stat().st_size != state.get('bytes')
                or state.get('documents', 0) <= 0):
            raise ValueError(f'{source}: 已完成源文件缺失或大小与manifest不符')
        print(f'复用已完成源 {source}: {state["documents"]} documents', flush=True)
        return state
    quarantine(final)
    dataset = (loader or load_hf_source)(source, cache_dir=cache_dir, num_proc=num_proc, offline=offline)
    expected_count = len(dataset)
    if expected_count <= 0:
        raise ValueError(f'{source}: HF数据集为空，不发布成功文件')
    partial = root/f'{source}.jsonl.partial'
    count = 0
    with partial.open('w', encoding='utf-8') as stream:
        for index, item in enumerate(dataset):
            stream.write(json.dumps(convert_document(item, source, index), ensure_ascii=False)+'\n')
            count += 1
            if count % 100000 == 0:
                print(f'{source}: {count}/{expected_count}', flush=True)
        if count != expected_count:
            raise ValueError(f'{source}: 转换行数 {count} != HF数据集完整行数 {expected_count}')
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(final)
    state = {'status': 'complete', 'format_version': FORMAT_VERSION, 'source': source,
             **expected_source, 'split': 'train', 'documents': count, 'expected_documents': expected_count,
             'bytes': final.stat().st_size, 'file': final.name,
             'expected_source_shards': len(EXPECTED_CHUNKS[source])}
    atomic_json(manifest_path, state)
    print(f'完成源 {source}: {count} documents', flush=True)
    return state


def prepare_corpus(output_dir, cache_dir=None, num_proc=8, offline=False, loader=None):
    if num_proc <= 0:
        raise ValueError('num_proc 必须为正数')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    combined = root/'medical_combined.jsonl'
    manifest_path = root/'medical_combined.manifest.json'
    # 作者旧combined可能只有textbooks。无本脚本有效manifest的文件不当作成功结果。
    old_state = None
    if manifest_path.exists():
        try:
            old_state = json.loads(manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            old_state = None
    valid_old = (isinstance(old_state, dict) and old_state.get('status') == 'complete'
                 and old_state.get('format_version') == FORMAT_VERSION
                 and old_state.get('source_revisions') == {n: s['revision'] for n, s in SOURCES.items()}
                 and set(old_state.get('counts', {})) == set(SOURCES)
                 and combined.is_file() and combined.stat().st_size == old_state.get('bytes'))
    if not valid_old:
        quarantine(combined)
        quarantine(manifest_path)
    states = {}
    for source in SOURCES:
        states[source] = prepare_source(root, source, cache_dir, num_proc, offline, loader)
    # 仅所有源都成功后进入合并。边复制边检查实际行数及原格式，不计算哈希。
    partial = root/'medical_combined.jsonl.partial'
    counts = {}
    with partial.open('w', encoding='utf-8') as destination:
        for source, state in states.items():
            count = 0
            with (root/state['file']).open(encoding='utf-8') as stream:
                for line in stream:
                    document = json.loads(line)
                    if (set(document) != {'id', 'text', 'source'} or document['source'] != source
                            or not isinstance(document['text'], str) or not document['text'].strip()):
                        raise ValueError(f'{source}: 已完成源含无效转换行 {count}')
                    destination.write(line if line.endswith('\n') else line+'\n')
                    count += 1
            if count != state['documents'] or count <= 0:
                raise ValueError(f'{source}: 实际行数 {count} 与已完成manifest不符')
            counts[source] = count
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(combined)
    state = {'status': 'complete', 'format_version': FORMAT_VERSION,
             'source_revisions': {n: s['revision'] for n, s in SOURCES.items()},
             'counts': counts, 'total_documents': sum(counts.values()),
             'bytes': combined.stat().st_size, 'file': combined.name, 'sources': states}
    atomic_json(manifest_path, state)
    print(f'完整双源语料已发布: {combined}; 实际行数 {state["total_documents"]}', flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', '--output_dir', type=Path, required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--num-proc', type=int, default=8)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    prepare_corpus(args.output_dir, args.cache_dir, args.num_proc, args.offline)


if __name__ == '__main__':
    main()
