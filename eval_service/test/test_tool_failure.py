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
