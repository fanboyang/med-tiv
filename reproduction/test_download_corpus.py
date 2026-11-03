"""严格 MedRAG 下载转换的最小离线测试。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def module():
    path = Path(__file__).with_name('download_corpus.py')
    if not path.exists():
        raise AssertionError('download_corpus.py 尚未实现')
    spec = importlib.util.spec_from_file_location('download_corpus', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class DownloadCorpusTests(unittest.TestCase):
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
