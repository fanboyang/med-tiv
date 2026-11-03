"""数据适配的最小离线检查；运行 python -m unittest discover -s reproduction -p test_prepare_data.py。"""
import importlib.util
import json
from pathlib import Path
import unittest


def module():
    path = Path(__file__).with_name('prepare_data.py')
    if not path.exists():
        raise AssertionError('prepare_data.py 尚未实现')
    spec = importlib.util.spec_from_file_location('prepare_data', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(i=1, question=None):
    return {'question': question or f'Question {i}?', 'question_id': i,
            'options': ['choice one', 'choice two'], 'correct_answer': 'SECRET_REFERENCE',
            'data_source': 'medqa', 'solutions': [
                {'solution': 'Candidate argues for B.', 'orm_label': 0,
                 'prm_hard_label': [1, 1]},
                {'solution': 'Candidate argues for A.', 'orm_label': 1,
                 'prm_hard_label': [0, 0]}]}


class PrepareDataTests(unittest.TestCase):
    def test_trace_label_uses_orm_not_steps_and_no_reference_leak(self):
        m = module()
        records = list(m.medprm_records([fixture()]))
        self.assertEqual([r['reward_model']['ground_truth'] for r in records], ['0', '1'])
        self.assertEqual(set(records[0]), {'prompt', 'reward_model', 'data_source', 'extra_info'})
        prompt = json.dumps(records[0]['prompt'])
        self.assertIn('Candidate argues for B.', prompt)
        for forbidden in ['SECRET_REFERENCE', 'prm_hard_label', 'orm_label', 'correct_answer']:
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(records[0]['prompt'][0]['role'], 'user')

    def test_empty_solutions_contribute_no_traces_but_missing_schema_fails(self):
        m = module()
        row = fixture()
        row['solutions'] = []
        self.assertEqual(list(m.medprm_records([row])), [])
        del row['solutions']
        with self.assertRaises(ValueError):
            list(m.medprm_records([row]))

    def test_trace_ids_are_unique_and_stable(self):
        m = module()
        records = list(m.medprm_records([fixture(1), fixture(2)]))
        ids = [r['extra_info'].get('id') for r in records]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [r['extra_info']['id'] for r in m.medprm_records([fixture(1), fixture(2)])])

    def test_missing_or_nonbinary_label_is_explicit_error(self):
        m = module()
        for bad in [None, -1, 2, '1', True]:
            row = fixture()
            row['solutions'][0]['orm_label'] = bad
            with self.assertRaises(ValueError):
                list(m.medprm_records([row]))
        row = fixture()
        del row['solutions'][0]['orm_label']
        with self.assertRaises(ValueError):
            list(m.medprm_records([row]))

    def test_question_grouping_ignores_different_ids_and_whitespace(self):
        m = module()
        rows = [fixture(i) for i in range(20)]
        rows.append(fixture(999, '  QUESTION   1?  '))
        train, valid = m.split_by_question(list(m.medprm_records(rows)), .2, 42)
        train_q = {r['extra_info']['question_key'] for r in train}
        valid_q = {r['extra_info']['question_key'] for r in valid}
        self.assertTrue(train and valid)
        self.assertFalse(train_q & valid_q)
        self.assertEqual(len(train) + len(valid), 42)
        self.assertEqual((train, valid), m.split_by_question(list(m.medprm_records(rows)), .2, 42))

    def test_evaluation_questions_are_excluded_before_training(self):
        m = module()
        rows = list(m.medprm_records([fixture(1), fixture(2)], excluded_questions={' question 1? '}))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['extra_info']['question_id'], '2')
        with self.assertRaises(ValueError):
            list(m.medprm_records([fixture()], source_split='test'))

    def test_evaluation_normalizers_and_gold_bounds(self):
        m = module()
        cases = [
            ('medqa', {'question': 'q', 'options': {'A': 'a', 'B': 'b'}, 'answer_idx': 'B'}, 'B'),
            ('medmcqa', {'question': 'q', 'opa': 'a', 'opb': 'b', 'opc': 'c', 'opd': 'd', 'cop': 0}, 'A'),
            ('mmlu_med', {'question': 'q', 'choices': ['a', 'b'], 'answer': 1}, 'B'),
            ('medxpertqa', {'question': 'q\nAnswer Choices: (A) a (B) b', 'options': {'A': 'a', 'B': 'b'}, 'label': 'A'}, 'A')]
        for name, row, gold in cases:
            normalized = m.normalize_evaluation(name, row, 0, 'test')
            self.assertEqual(normalized['gold'], gold)
            self.assertEqual(normalized['question'], 'q')
            self.assertEqual(list(normalized['options']), ['A', 'B'] if name != 'medmcqa' else list('ABCD'))
        with self.assertRaises(ValueError):
            m.normalize_evaluation('medmcqa', {**cases[1][1], 'cop': -1}, 0, 'validation')

    def test_balanced_budget_is_exact_and_explicit_when_insufficient(self):
        m = module()
        records = list(m.medprm_records([fixture(i) for i in range(4)]))
        selected = m.balanced_sample(records, 6, 42)
        self.assertEqual([r['reward_model']['ground_truth'] for r in selected].count('1'), 3)
        with self.assertRaises(ValueError):
            m.balanced_sample(records, 10, 42)
        with self.assertRaises(ValueError):
            m.balanced_sample(records, 3, 42)


if __name__ == '__main__':
    unittest.main()
