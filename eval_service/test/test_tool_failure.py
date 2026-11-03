"""Real client methods/API routes with model construction and vLLM launch mocked."""
import asyncio
from contextlib import contextmanager
import importlib
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

SERVICE_ROOT=str(Path(__file__).resolve().parents[1])
if SERVICE_ROOT not in sys.path: sys.path.insert(0,SERVICE_ROOT)
# SamplingParams is not used by the code under test; no CUDA/vLLM is needed.
fake_vllm=ModuleType('vllm')
fake_vllm.SamplingParams=object
with patch.dict(sys.modules,{'vllm':fake_vllm}):
    model_service=importlib.import_module('model_service')
    with patch('logging.FileHandler',return_value=logging.NullHandler()):
        app_module=importlib.import_module('app')
from fastapi.testclient import TestClient

class AsyncResponse:
    async def __aenter__(self): raise RuntimeError('retriever offline')
    async def __aexit__(self,*_): return False

class ToolFailureTests(unittest.TestCase):
    def service(self):
        service=model_service.ModelService.__new__(model_service.ModelService)
        service.tool_config=SimpleNamespace(tool_server_url='http://127.0.0.1:1')
        service.session=SimpleNamespace(post=lambda *a,**k:AsyncResponse(),close=AsyncMock())
        service.vllm_processes=[]
        return service

    def test_sync_transport_failure_raises_instead_of_done(self):
        with patch.object(model_service.requests,'post',side_effect=RuntimeError('offline')):
            with self.assertRaises(model_service.ToolServerError):
                self.service().call_tool_server(['x'],['<search>x</search>'],[False])

    def test_async_transport_failure_raises_instead_of_done(self):
        with self.assertRaises(model_service.ToolServerError):
            asyncio.run(self.service().call_tool_server_async(['x'],['<search>x</search>'],[False]))

    def test_malformed_tool_response_is_explicit_failure(self):
        response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'observations':[]})
        with patch.object(model_service.requests,'post',return_value=response):
            with self.assertRaises(model_service.ToolServerError):
                self.service().call_tool_server(['x'],['action'],[False])

    def test_generation_keeps_model_text_separate_from_truncated_observations(self):
        service=self.service()
        service.clients=[object()]
        service.model_config=SimpleNamespace(model='model',max_model_len=16384)
        service.tool_config.max_turns=2
        service.tool_config.max_response_length=8192
        service.tokenizer=SimpleNamespace(encode=lambda text,**kwargs:list(map(ord,text)),
            decode=lambda ids,**kwargs:''.join(map(chr,ids)))
        first='<think>look</think><search>q</search>'
        final='<think>done</think><answer>1</answer>'
        service.send_request=AsyncMock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(token_ids=list(map(ord,first)))]),
                                                  SimpleNamespace(choices=[SimpleNamespace(token_ids=list(map(ord,final)))])])
        truncated='<answer>0</answer></information>'
        service._postprocess_responses=AsyncMock(side_effect=[([first],[False],['stop'],[None]),
                                                             ([final],[True],['stop'],[None])])
        service.call_tool_server_async=AsyncMock(side_effect=[{'observations':[truncated],'dones':[False],'valids':[True]},
                                                              {'observations':[''],'dones':[True],'valids':[False]}])
        service.post_process_observations=AsyncMock(side_effect=[[truncated],['']])
        values=asyncio.run(service.generate_with_tools(['prompt'],{'max_tokens':1024,'stop':['</search>','</answer>']}))
        self.assertEqual(values[0],[first+truncated+final])
        self.assertEqual(values[3],[first+final])
        self.assertIn(truncated,service.tokenizer.decode(service.send_request.await_args_list[1].args[1][0]))

    def budget_service(self,prompt_budget=100,response_budget=6):
        service=self.service()
        service.clients=[object()]
        service.model_config=SimpleNamespace(model='model',max_model_len=prompt_budget)
        service.tool_config.max_turns=4
        service.tool_config.max_response_length=response_budget
        service.tokenizer=SimpleNamespace(encode=lambda text,**kwargs:list(map(ord,text)),
            decode=lambda ids,**kwargs:''.join(map(chr,ids)))
        return service

    def test_response_budget_caps_observations_and_stops(self):
        service=self.budget_service(response_budget=6)
        service.send_request=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(token_ids=list(map(ord,'ABC')))]))
        service._postprocess_responses=AsyncMock(return_value=(['ABC'],[False],['stop'],[None]))
        service.call_tool_server_async=AsyncMock(return_value={'observations':['0123456789'],'dones':[False],'valids':[True]})
        service.post_process_observations=AsyncMock(return_value=['0123456789'])
        result=asyncio.run(service.generate_with_tools(['q'],{'max_tokens':4,'stop':[]}))
        self.assertEqual(result[0],['ABC012'])
        self.assertEqual(result[3],['ABC'])
        self.assertEqual(result[4],[6])
        self.assertEqual(service.send_request.await_count,1)
        self.assertEqual(service.send_request.await_args.args[3]['max_tokens'],4)

    def test_next_action_uses_remaining_total_budget(self):
        service=self.budget_service(response_budget=7)
        service.send_request=AsyncMock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(token_ids=[65,66,67])]),
                                                  SimpleNamespace(choices=[SimpleNamespace(token_ids=[68,69,70])])])
        service._postprocess_responses=AsyncMock(side_effect=[(['ABC'],[False],['stop'],[None]),
                                                             (['DEF'],[True],['stop'],[None])])
        service.call_tool_server_async=AsyncMock(side_effect=[{'observations':['x'],'dones':[False],'valids':[True]},
                                                              {'observations':[''],'dones':[True],'valids':[False]}])
        service.post_process_observations=AsyncMock(side_effect=[['x'],['']])
        result=asyncio.run(service.generate_with_tools(['q'],{'max_tokens':4,'stop':[]}))
        self.assertEqual([call.args[3]['max_tokens'] for call in service.send_request.await_args_list],[4,3])
        self.assertEqual(result[0],['ABCxDEF'])
        self.assertEqual(result[3],['ABCDEF'])
        self.assertEqual(result[4],[7])

    def test_multiple_prompts_have_independent_context_budgets(self):
        service=self.budget_service(prompt_budget=8,response_budget=8)
        # A model-only 2-token action after a 6-token prompt hits the context cap.
        service.send_request=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(token_ids=list(map(ord,'AB')))]))
        service._postprocess_responses=AsyncMock(return_value=(['AB'],[True],['stop'],[None]))
        service.call_tool_server_async=AsyncMock(return_value={'observations':[''],'dones':[True],'valids':[False]})
        service.post_process_observations=AsyncMock(return_value=[''])
        asyncio.run(service.generate_with_tools(['123456','q'],{'max_tokens':4,'stop':[]}))
        self.assertEqual([call.args[3]['max_tokens'] for call in service.send_request.await_args_list],[2,4])

    def test_missing_original_token_ids_is_explicit_failure(self):
        service=self.budget_service()
        service.send_request=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(text='AB')]))
        service._postprocess_responses=AsyncMock(return_value=(['AB'],[True],['stop'],[None]))
        with self.assertRaises(ValueError):
            asyncio.run(service.generate_with_tools(['q'],{'max_tokens':4,'stop':[]}))

    def test_both_api_response_types_expose_generated_text(self):
        service=self.service()
        service.model_config=SimpleNamespace(model='model',max_model_len=16384)
        service.tool_config.action_stop_tokens=['</search>','</answer>']
        service.encode_lock=asyncio.Lock()
        service.tokenizer=SimpleNamespace(encode=lambda text,**kwargs:list(text),apply_chat_template=lambda *a,**k:'prompt')
        service.generate_with_tools=AsyncMock(return_value=(['full with observation'],['stop'],[None],['model only'],[21]))
        for method,body in [(service.completions_async,{'model':'model','prompt':'prompt'}),
                            (service.chat_completions_async,{'model':'model','messages':[{'role':'user','content':'q'}]})]:
            result=asyncio.run(method(body))
            self.assertEqual(result['choices'][0]['generated_text'],'model only')
            self.assertEqual(result['choices'][0]['response_token_count'],21)

    def test_tool_failure_returns_503_for_both_api_routes(self):
        service=SimpleNamespace(load_model=lambda:None,
            completions_async=AsyncMock(side_effect=model_service.ToolServerError('offline')),
            chat_completions_async=AsyncMock(side_effect=model_service.ToolServerError('offline')))
        with patch.object(app_module,'ModelService',return_value=service):
            app=app_module.create_app(SimpleNamespace(),SimpleNamespace(),SimpleNamespace())
        with TestClient(app) as client:
            for endpoint in ['/completions','/chat/completions']:
                response=client.post(endpoint,json={'model':'x','prompt':'q'})
                self.assertEqual(response.status_code,503)
                self.assertEqual(response.json()['detail'],'Tool server unavailable')

if __name__=='__main__': unittest.main()
