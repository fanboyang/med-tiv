import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from reproduction.curriculum import informative, choose_balanced, validate_pool, check_length, sample_group
from reproduction.evaluate import Journal

class FakeTokenizer:
    def encode(self,text,**kwargs): return list(range(len(text)))
    def apply_chat_template(self,messages,**kwargs): return messages[0]['content']

class CurriculumTests(unittest.TestCase):
    def test_reward_variation_not_label_variation(self):
        self.assertFalse(informative([0.0] * 8))
        self.assertFalse(informative([1.0] * 8))
        self.assertTrue(informative([1.0] * 7 + [0.0]))
        self.assertTrue(informative([0.25] * 7 + [1.0]))

    def test_balance_and_no_replacement(self):
        rows = [{'id':str(i),'label':i%2,'rewards':[0,1]*4} for i in range(8)]
        selected = choose_balanced(rows,4)
        self.assertEqual(len({r['id'] for r in selected}),4)
        self.assertEqual(sum(r['label'] for r in selected),2)

    def test_shortage_fails(self):
        with self.assertRaises(ValueError):
            choose_balanced([{'id':'x','label':0,'rewards':[0,1]*4}],4)

    def test_stable_ids_and_input_identity(self):
        row={'prompt':[{'role':'user','content':'q'}], 'reward_model':{'ground_truth':'0'},
             'extra_info':{'id':'source:123'}}
        pool=validate_pool([row])
        self.assertEqual(pool[0]['id'],'source:123')
        with self.assertRaises(ValueError): validate_pool([row,row])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'requests.jsonl'
            Journal(path,{'pool':pool})
            row['prompt'][0]['content']='changed'
            with self.assertRaises(ValueError): Journal(path,{'pool':validate_pool([row])})

    def test_prompt_and_context_boundaries(self):
        tokenizer=FakeTokenizer()
        check_length(tokenizer,'1234',4,2,6)
        with self.assertRaises(ValueError):check_length(tokenizer,'12345',4,2,7)
        with self.assertRaises(ValueError):check_length(tokenizer,'1234',4,3,6)

    def test_partial_transport_resume_reuses_completed_requests(self):
        tokenizer=FakeTokenizer()
        row={'id':'real-id','label':1,'messages':[{'role':'user','content':'q'}]}
        raw={'choices':[{'text':'<think>x</think><answer>1</answer>',
                         'generated_text':'<think>x</think><answer>1</answer>', 'response_token_count':33}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'requests.jsonl'
            journal=Journal(path,{})
            with patch('reproduction.evaluate.post',side_effect=[raw,URLError('offline')]):
                with self.assertRaises(RuntimeError):sample_group(row,0,tokenizer,journal,'model','local',42,1024,8192,16384,8192)
            self.assertEqual(len(path.read_text().splitlines()),3)
            with patch('reproduction.evaluate.post',return_value=raw) as request:
                result=sample_group(row,0,tokenizer,Journal(path,{}),'model','local',42,1024,8192,16384,8192)
                self.assertEqual(request.call_count,7)
                self.assertEqual(result['rewards'],[1.0]*8)

    def test_truncated_observation_does_not_pollute_model_only_reward(self):
        row={'id':'real-id','label':1,'messages':[{'role':'user','content':'q'}]}
        text='<think>x</think><answer>1</answer>'
        raw={'choices':[{'text':text+'<answer>0</answer></information>', 'generated_text':text,'response_token_count':64}]}
        with tempfile.TemporaryDirectory() as directory:
            with patch('reproduction.evaluate.post',return_value=raw):
                result=sample_group(row,0,FakeTokenizer(),Journal(Path(directory)/'requests.jsonl',{}),
                                    'model','local',42,1024,8192,16384,8192)
            self.assertEqual(result['rewards'],[1.0]*8)

    def test_malformed_response_is_terminal_and_not_resampled(self):
        row={'id':'real-id','label':1,'messages':[{'role':'user','content':'q'}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'requests.jsonl'
            with patch('reproduction.evaluate.post',return_value={'choices':[{'text':'<answer>1</answer>'}]}):
                with self.assertRaises(RuntimeError):sample_group(row,0,FakeTokenizer(),Journal(path,{}),'model','local',42,1024,8192,16384,8192)
            with patch('reproduction.evaluate.post') as request:
                with self.assertRaises(RuntimeError):sample_group(row,0,FakeTokenizer(),Journal(path,{}),'model','local',42,1024,8192,16384,8192)
                request.assert_not_called()

if __name__ == '__main__': unittest.main()
