"""严格 MedRAG 下载转换的最小离线测试。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def module():
    path = Path(__file__).with_name('download_corpus.py')
    if not path.exists():
        raise AssertionError('download_corpus.py 尚未实现')
    spec = importlib.util.spec_from_file_location('download_corpus', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class DownloadCorpusTests(unittest.TestCase):
    def test_hf_metadata_and_fixed_revision_cache_avoid_tree_pagination(self):
        m = module()
        revision = m.SOURCES['pubmed']['revision']
        filenames = [f'chunk/pubmed23n{i:04d}.jsonl' for i in range(1, 1167)]
        metadata = SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=f, size=3) for f in filenames])
        with tempfile.TemporaryDirectory() as temp:
            cached = Path(temp)/'cached.jsonl'
            cached.write_text('{}\n')
            with patch.dict('os.environ', {'HF_ENDPOINT': 'https://hf-mirror.com'}), \
                 patch('huggingface_hub.HfApi') as api, \
                 patch('huggingface_hub.try_to_load_from_cache', return_value=str(cached)) as lookup, \
                 patch('huggingface_hub.hf_hub_download') as download:
                api.return_value.dataset_info.return_value = metadata
                paths = m.hf_local_files('pubmed', num_proc=2)
                self.assertEqual(len(paths), 1166)
                api.assert_called_once_with(endpoint='https://hf-mirror.com')
                api.return_value.dataset_info.assert_called_once_with('MedRAG/pubmed', revision=revision, files_metadata=True)
                self.assertEqual(lookup.call_count, 1166)
                self.assertEqual(lookup.call_args.kwargs['revision'], revision)
                download.assert_not_called()
                # 缺片只恢复该固定revision文件，仍使用默认HF_HOME下hub缓存。
                missing = filenames[-1]
                lookup.side_effect = lambda *a, **kw: None if kw['filename'] == missing else str(cached)
                download.return_value = str(cached)
                self.assertEqual(len(m.hf_local_files('pubmed', num_proc=2)), 1166)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.kwargs['revision'], revision)
                self.assertEqual(download.call_args.kwargs['endpoint'], 'https://hf-mirror.com')
                self.assertNotIn('cache_dir', download.call_args.kwargs)
                download.side_effect = RuntimeError('missing shard transfer failed')
                with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
                    m.hf_local_files('pubmed', num_proc=2)

    def test_missing_shard_metadata_or_revision_mismatch_fails(self):
        m = module()
        revision = m.SOURCES['pubmed']['revision']
        full = [SimpleNamespace(rfilename=f'chunk/pubmed23n{i:04d}.jsonl') for i in range(1, 1167)]
        renamed = full[:-1]+[SimpleNamespace(rfilename='chunk/unexpected.jsonl')]
        for metadata in [SimpleNamespace(sha=revision, siblings=full[:-1]),
                         SimpleNamespace(sha=revision, siblings=renamed),
                         SimpleNamespace(sha='another-revision', siblings=full)]:
            with patch('huggingface_hub.HfApi') as api:
                api.return_value.dataset_info.return_value = metadata
                with self.assertRaises(ValueError):
                    m.hf_local_files('pubmed', num_proc=1)

    def test_official_zero_byte_shard_is_not_treated_as_corruption(self):
        m = module()
        revision = m.SOURCES['pubmed']['revision']
        names = [f'chunk/pubmed23n{i:04d}.jsonl' for i in range(1, 1167)]
        empty_name = 'chunk/pubmed23n0654.jsonl'
        siblings = [SimpleNamespace(rfilename=f, size=0 if f == empty_name else 3) for f in names]
        with tempfile.TemporaryDirectory() as temp:
            valid, empty = Path(temp)/'valid.jsonl', Path(temp)/'empty.jsonl'
            valid.write_text('{}\n')
            empty.write_text('')
            with patch('huggingface_hub.HfApi') as api, \
                 patch('huggingface_hub.try_to_load_from_cache') as lookup:
                api.return_value.dataset_info.return_value = SimpleNamespace(sha=revision, siblings=siblings)
                lookup.side_effect = lambda *a, **kw: str(empty if kw['filename'] == empty_name else valid)
                paths = m.hf_local_files('pubmed', num_proc=2)
                self.assertEqual(len(paths), 1166)
                self.assertIn(str(empty), paths)
                siblings[653].size = 1
                with self.assertRaisesRegex(ValueError, 'size'):
                    m.hf_local_files('pubmed', num_proc=2)

    def test_dataset_loader_uses_only_explicit_local_json_paths(self):
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'shard.jsonl'
            path.write_text('{}\n')
            with patch.object(m, 'hf_local_files', return_value=[str(path)]), \
                 patch('datasets.load_dataset', return_value='dataset') as load:
                self.assertEqual(m.load_hf_source('pubmed', cache_dir='/dataset-cache', num_proc=2), 'dataset')
                self.assertEqual(load.call_args.args, ('json',))
                self.assertEqual(load.call_args.kwargs['data_files'], {'train': [str(path)]})
                self.assertNotIn('revision', load.call_args.kwargs)

    def test_json_builder_preserves_legal_empty_file_without_fake_records(self):
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            empty, valid = root/'empty.jsonl', root/'valid.jsonl'
            empty.write_text('')
            valid.write_text('{"id":"x","title":"title","content":"body"}\n')
            # HF JSON builder不能以空文件推断schema；保留两条路径，调整顺序/worker分组。
            with patch.object(m, 'hf_local_files', return_value=[str(empty), str(valid)]):
                dataset = m.load_hf_source('pubmed', cache_dir=str(root/'cache'), num_proc=2)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]['id'], 'x')
            self.assertEqual(dataset[0]['content'], 'body')
            # 8 workers的边界：16片含1空片，每个分组至少2片，首片均非空。
            valid_paths = []
            for i in range(15):
                shard = root/f'valid-{i}.jsonl'
                shard.write_text('{"id":"x","content":"body"}\n')
                valid_paths.append(str(shard))
            with patch.object(m, 'hf_local_files', return_value=[str(empty)]+valid_paths), \
                 patch('datasets.load_dataset', return_value='dataset') as load:
                m.load_hf_source('pubmed', cache_dir=str(root/'cache'), num_proc=8)
                self.assertEqual(load.call_args.kwargs['num_proc'], 8)
                self.assertEqual(load.call_args.kwargs['data_files']['train'], valid_paths+[str(empty)])

    def test_text_and_title_follow_author_format(self):
        m = module()
        self.assertEqual(m.convert_document({'id': 'x', 'title': 'Title', 'content': 'Body'}, 'pubmed', 0),
                         {'id': 'x', 'text': 'Title. Body', 'source': 'pubmed'})
        self.assertEqual(m.convert_document({'title': 'Title', 'text': 'Title in body'}, 'textbooks', 2)['text'], 'Title in body')
        with self.assertRaises(ValueError):
            m.convert_document({'title': 'Title', 'content': ''}, 'pubmed', 0)
        with self.assertRaises(ValueError):
            m.convert_document({'content': None}, 'pubmed', 0)

    def test_missing_pubmed_never_publishes_combined(self):
        m = module()
        def load(name, **kwargs):
            if name == 'pubmed':
                raise RuntimeError('one shard download failed')
            return [{'id': 't', 'content': 'textbook body'}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # 模拟作者脚本此前错误发布的仅textbooks文件。
            (root/'medical_combined.jsonl').write_text('{}\n')
            with self.assertRaises(RuntimeError):
                m.prepare_corpus(root, loader=load)
            self.assertFalse((root/'medical_combined.jsonl').exists())
            self.assertFalse((root/'medical_combined.manifest.json').exists())
            self.assertTrue((root/'textbooks.jsonl').exists())

    def test_completed_source_reused_and_actual_counts_recorded(self):
        m = module()
        calls = []
        def load(name, **kwargs):
            calls.append(name)
            return [{'id': name, 'content': name+' body'}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m.prepare_corpus(root, loader=load)
            m.prepare_corpus(root, loader=load)
            self.assertEqual(calls, ['textbooks', 'pubmed'])
            manifest = json.loads((root/'medical_combined.manifest.json').read_text())
            self.assertEqual(manifest['total_documents'], 2)
            self.assertEqual(manifest['counts'], {'textbooks': 1, 'pubmed': 1})
            rows = [json.loads(x) for x in (root/'medical_combined.jsonl').read_text().splitlines()]
            self.assertEqual({r['source'] for r in rows}, {'textbooks', 'pubmed'})

    def test_incomplete_source_stays_partial_and_cannot_resume_as_complete(self):
        m = module()
        class BrokenRows:
            def __len__(self):
                return 2
            def __iter__(self):
                yield {'content': 'first document'}
                raise RuntimeError('second shard unavailable')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                m.prepare_source(root, 'pubmed', loader=lambda *a, **k: BrokenRows())
            self.assertTrue((root/'pubmed.jsonl.partial').exists())
            self.assertFalse((root/'pubmed.jsonl').exists())
            self.assertFalse((root/'pubmed.manifest.json').exists())
            state = m.prepare_source(root, 'pubmed', loader=lambda *a, **k: [{'content': 'complete'}])
            self.assertEqual(state['documents'], 1)
            self.assertFalse((root/'pubmed.jsonl.partial').exists())

    def test_truncated_iteration_fails_expected_length_check(self):
        m = module()
        class ShortRows:
            def __len__(self):
                return 2
            def __iter__(self):
                yield {'content': 'first document'}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                m.prepare_source(Path(temp), 'pubmed', loader=lambda *a, **k: ShortRows())
            self.assertFalse((Path(temp)/'pubmed.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
