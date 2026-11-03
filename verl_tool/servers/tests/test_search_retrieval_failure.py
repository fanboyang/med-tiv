"""Exercise real search/worker/API failure propagation without a retrieval model."""
import asyncio
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import requests
from fastapi.testclient import TestClient
from verl_tool.servers.tools import search_retrieval as search
try:
    import fire
except ImportError:
    with patch.dict(sys.modules,{'fire':ModuleType('fire')}):
        server_module=importlib.import_module('verl_tool.servers.tool_server')
else:
    server_module=importlib.import_module('verl_tool.servers.tool_server')

class RetrievalFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(search,'setup_debug_logger',return_value=str(Path(self.temp.name)/'debug.log')):
            self.tool=search.SearchRetrievalTool()

    def test_network_failure_raises_and_is_not_cached(self):
        with patch.object(self.tool,'_get_session',return_value=SimpleNamespace(post=lambda *a,**k:(_ for _ in ()).throw(requests.ConnectionError('offline')))), patch.object(search.time,'sleep'):
            with self.assertRaises(search.RetrievalServiceError):
                self.tool.conduct_action('t','<search>x</search>',{})
        self.assertFalse(self.tool.query_cache)
        self.assertFalse(self.tool.env_cache)

    def test_protocol_error_is_not_empty_result(self):
        for raw in [{}, {'result':[]}, {'result':[[{'document':{}}]]}]:
            response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:raw)
            with self.subTest(raw=raw), patch.object(self.tool,'_get_session',return_value=SimpleNamespace(post=lambda *a,**k:response)):
                with self.assertRaises(search.RetrievalServiceError):self.tool._batch_search(['q'])

    def test_valid_empty_and_invalid_action_preserve_semantics(self):
        response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'result':[[]]})
        with patch.object(self.tool,'_get_session',return_value=SimpleNamespace(post=lambda *a,**k:response)):
            obs,done,valid=self.tool.conduct_action('empty','<search>x</search>',{})
            self.assertEqual(obs.strip(),'<information></information>')
            self.assertFalse(done)
            self.assertTrue(valid)
        self.assertEqual(self.tool.conduct_action('invalid','not a tool',{}),('',False,False))
        self.assertEqual(self.tool.conduct_action('done','<answer>0</answer>',{}),('',True,False))

    def test_worker_does_not_package_retrieval_failure_as_observation(self):
        manager=server_module.AsyncToolManager.__new__(server_module.AsyncToolManager)
        async def fail(): raise search.RetrievalServiceError('offline')
        with self.assertRaises(search.RetrievalServiceError):
            asyncio.run(manager._execute_tool_tasks([('search_retrieval',[0],fail())],[None],[False],[False]))

    def test_api_returns_503_not_observation_text(self):
        manager=SimpleNamespace(process_actions=AsyncMock(side_effect=search.RetrievalServiceError('offline')))
        with patch.object(server_module,'AsyncToolManager',return_value=manager):
            server=server_module.AsyncToolServer(('search_retrieval',),server_module.ServerConfig(enable_hashing=False))
        client=TestClient(server.app)
        response=client.post('/get_observation',json={'trajectory_ids':['t'],'actions':['<search>x</search>'],'extra_fields':[{}]})
        self.assertEqual(response.status_code,503)
        self.assertNotIn('observations',response.json())
        client.close()

if __name__=='__main__':unittest.main()
