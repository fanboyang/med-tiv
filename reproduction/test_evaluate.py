import unittest
import json
from io import BytesIO
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from reproduction.evaluate import generator_messages, parse_answer, parse_verdict, hard_vote, summarize, Journal, ResponseDecodeError

class EvaluationTests(unittest.TestCase):
    def test_gold_is_not_in_prompt(self):
        row={'id':'x','question':'Pick','options':{'A':'alpha','B':'beta'},'gold':'SECRET'}
        self.assertNotIn('SECRET', str(generator_messages(row)))
    def test_strict_parsing(self):
        self.assertEqual(parse_answer('<answer>B</answer>', ['A','B']), 'B')
        self.assertIsNone(parse_answer('I think B', ['A','B']))
        self.assertIsNone(parse_verdict('0 then 1'))
        self.assertIsNone(parse_verdict('<answer>2</answer>'))
        self.assertEqual(parse_verdict('<think>x</think><answer>1</answer>'), 1)
        self.assertEqual(parse_verdict('<answer>0</answer><answer>1</answer>'), 1)
        self.assertIsNone(parse_verdict('<answer>1</answer><answer>2</answer>'))
    def test_verdict_ignores_retrieval_answer_and_rejects_unpaired_observation(self):
        leaked='<answer>0</answer><information>quoted <answer>1</answer></information>'
        self.assertEqual(parse_verdict(leaked),0)
        self.assertIsNone(parse_verdict('<tool_response><answer>1</answer></tool_response>'))
        with self.assertRaises(ValueError):parse_verdict('<answer>1</answer><information>truncated')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'journal.jsonl'
            raw={'choices':[{'text':'<answer>1</answer><information>truncated'}]}
            with patch('reproduction.evaluate.post',return_value=raw):
                self.assertIsNone(Journal(path,{}).call(['bad'],'local',{},lambda x:parse_verdict(x['choices'][0]['text']),1))
            with patch('reproduction.evaluate.post') as request:
                self.assertIsNone(Journal(path,{}).call(['bad'],'local',{},lambda x:0,1))
                request.assert_not_called()

    def test_author_vote_and_ties(self):
        self.assertEqual(hard_vote([('B',0),('A',0)]), ('B',True))
        self.assertEqual(hard_vote([('B',0),('A',1)]), ('A',False))
        self.assertEqual(hard_vote([('B',1),(None,1)]), (None,False))
    def test_missing_stays_in_denominator(self):
        s=summarize([{'prediction':'A','gold':'A','all_rejected':True}, {'prediction':None,'gold':'B','all_rejected':False}])
        self.assertEqual(s['accuracy'], .5)
        self.assertEqual(s['missing'],1)
        self.assertEqual(s['all_rejected'],1)

    def test_valid_response_parse_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'journal.jsonl'
            journal=Journal(path, {'model':'x'})
            with patch('reproduction.evaluate.post',return_value={'text':'bad'}):
                self.assertIsNone(journal.call(['a'],'local',{},lambda raw:None,1))
            self.assertIn('raw_response',path.read_text())
            resumed=Journal(path, {'model':'x'})
            with patch('reproduction.evaluate.post') as request:
                self.assertIsNone(resumed.call(['a'],'local',{},lambda raw:0,1))
                request.assert_not_called()
            with self.assertRaises(ValueError): Journal(path, {'model':'other'})

    def test_invalid_http_json_is_preserved_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'journal.jsonl'
            journal=Journal(path,{})
            with patch('reproduction.evaluate.post',side_effect=ResponseDecodeError('not json')):
                self.assertIsNone(journal.call(['badjson'],'local',{},lambda raw:0,1))
            self.assertIn('not json',path.read_text())
            with patch('reproduction.evaluate.post') as request:
                self.assertIsNone(Journal(path,{}).call(['badjson'],'local',{},lambda raw:0,1))
                request.assert_not_called()

    def test_transport_and_5xx_are_retryable_but_4xx_terminal(self):
        for failure,retry in [(URLError('offline'),True),
                              (HTTPError('local',503,'busy',{},BytesIO(b'service unavailable')),True),
                              (HTTPError('local',400,'bad request',{},BytesIO(b'invalid request')),False)]:
            with self.subTest(failure=type(failure),retry=retry), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'journal.jsonl'
                journal=Journal(path,{})
                with patch('reproduction.evaluate.post',side_effect=failure):
                    self.assertIsNone(journal.call(['a'],'local',{},lambda raw:0,1))
                failed=json.loads(path.read_text().splitlines()[1])
                if isinstance(failure,HTTPError):
                    self.assertEqual(failed['http_status'],failure.code)
                    self.assertEqual(failed['raw_response'],'service unavailable' if failure.code==503 else 'invalid request')
                resumed=Journal(path,{})
                with patch('reproduction.evaluate.post',return_value={'text':'0'}) as request:
                    value=resumed.call(['a'],'local',{},lambda raw:0,1)
                    self.assertEqual(value,0 if retry else None)
                    self.assertEqual(request.call_count,int(retry))
                with patch('reproduction.evaluate.post') as request:
                    self.assertEqual(resumed.call(['a'],'local',{},lambda raw:1,1),value)
                    request.assert_not_called()

if __name__=='__main__': unittest.main()
