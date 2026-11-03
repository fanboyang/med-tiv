"""Device/precision tests using tiny CPU tensors, without downloading models."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
import numpy as np
import torch

path=Path(__file__).resolve().parents[1]/'verl_tool/servers/tools/utils/retrieval_server.py'
spec=importlib.util.spec_from_file_location('retrieval_server_device_test',path)
retrieval=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)

class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    def forward(self,input_ids,attention_mask,return_dict=True):
        assert input_ids.device.type=='cpu'
        hidden=torch.tensor([[[3.,4.,0.],[0.,1.,0.]]]).repeat(input_ids.shape[0],1,1)
        return SimpleNamespace(pooler_output=None,last_hidden_state=hidden)

def tokenize(texts,**kwargs):
    return {'input_ids':torch.ones(len(texts),2,dtype=torch.long),
            'attention_mask':torch.ones(len(texts),2,dtype=torch.long)}

class RetrievalDeviceTests(unittest.TestCase):
    def test_gpu_defaults_are_unchanged(self):
        args=retrieval.build_parser().parse_args([])
        self.assertEqual((args.encoder_device,args.precision),('cuda','fp16'))
        config=retrieval.Config()
        self.assertEqual((config.encoder_device,config.precision,config.retrieval_use_fp16),('cuda','fp16',True))
        model=MagicMock();model.to.return_value=model
        with patch.object(retrieval.AutoModel,'from_pretrained',return_value=model),patch.object(retrieval.AutoTokenizer,'from_pretrained'):
            retrieval.load_model('unused',use_fp16=True)
        self.assertEqual(model.to.call_args.kwargs['device'],torch.device('cuda'))
        self.assertEqual(model.to.call_args.kwargs['dtype'],torch.float16)

    def test_cpu_encoder_and_full_flat_search_never_touch_cuda(self):
        index=SimpleNamespace(search=lambda values,k:(np.array([[.9]],dtype=np.float32),np.array([[0]])))
        config=retrieval.Config(retrieval_method='medcpt',faiss_gpu=False,encoder_device='cpu',precision='fp32',
                                retrieval_pooling_method='cls',retrieval_topk=1)
        with patch.object(retrieval.AutoModel,'from_pretrained',side_effect=lambda *a,**k:TinyEncoder()), \
             patch.object(retrieval.AutoTokenizer,'from_pretrained',return_value=tokenize), \
             patch.object(retrieval.faiss,'read_index',return_value=index), \
             patch.object(retrieval,'load_corpus',return_value=[{'id':'0','text':'evidence'}]), \
             patch.object(torch.Tensor,'cuda',side_effect=AssertionError('CPU called cuda')), \
             patch.object(torch.cuda,'empty_cache',side_effect=AssertionError('CPU called CUDA cache')):
            server=retrieval.DenseRetriever(config)
            embedding=server.encoder.encode(['q'])
            np.testing.assert_allclose(embedding,[[.6,.8,0]],atol=1e-7)
            results,scores=server.batch_search(['q'],return_score=True)
            self.assertEqual(results,[[{'id':'0','text':'evidence'}]])
            self.assertEqual(len(scores[0]),1)
            self.assertEqual(next(server.encoder.model.parameters()).device.type,'cpu')
            self.assertEqual(next(server.encoder.model.parameters()).dtype,torch.float32)

    def test_illegal_device_precision_and_faiss_combinations_fail(self):
        for kwargs in [{'encoder_device':'cpu','precision':'fp16','faiss_gpu':False},
                       {'encoder_device':'cpu','precision':'fp32','faiss_gpu':True},
                       {'encoder_device':'auto','precision':'fp32','faiss_gpu':False},
                       {'encoder_device':'cpu','precision':'bf16','faiss_gpu':False}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):retrieval.Config(**kwargs)
        with patch.object(retrieval.AutoModel,'from_pretrained') as loader:
            with self.assertRaises(ValueError):retrieval.load_model('unused',encoder_device='cpu',precision='fp16')
            loader.assert_not_called()

if __name__=='__main__':unittest.main()
