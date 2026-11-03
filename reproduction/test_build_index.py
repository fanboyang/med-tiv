"""仅检查生产语料解析函数；AST加载纯函数，避免导入GPU和FAISS依赖。"""
import ast
import json
from pathlib import Path
from typing import List
import unittest


def finalize_text(line, model_name='medcpt', line_number=7):
    source = Path(__file__).resolve().parents[1]/'build_index.py'
    tree = ast.parse(source.read_text())
    names = {'_maybe_add_prefix', '_finalize_text'}
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'json': json, 'List': List, 'model_name': model_name}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    function = namespace['_finalize_text']
    return function(line, line_number=line_number)


class BuildIndexTextTests(unittest.TestCase):
    def test_short_nonempty_text_remains_exact(self):
        for value in ['A', '中', '  short  ', 'long text unchanged']:
            self.assertEqual(finalize_text(json.dumps({'text': value, 'title': 'another longer title'})), value)
        self.assertEqual(finalize_text('{"text":"A"}', model_name='e5'), 'passage: A')

    def test_invalid_json_and_empty_or_nonstring_text_fail(self):
        for value in ['', '  ', '{broken', 'null', '[]', '"string"', '{}',
                      '{"text":""}', '{"text":"  ","title":"valid title"}',
                      '{"text":null}', '{"text":123}']:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'line 7'):
                    finalize_text(value)


if __name__ == '__main__':
    unittest.main()
