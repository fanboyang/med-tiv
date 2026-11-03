"""Dr.GRPO fixed-normalizer and actor accumulation regression tests (CPU torch).

Load the real agg_loss AST to avoid importing unrelated VeRL training dependencies.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


_HELPERS = {'torch': torch}
_load_functions(_ROOT / 'verl/verl/utils/torch_functional.py', {'masked_sum', 'masked_mean'}, _HELPERS)
_NAMESPACE = {'torch': torch, 'os': os, 'verl_F': SimpleNamespace(masked_mean=_HELPERS['masked_mean'])}
_load_functions(_ROOT / 'verl/verl/trainer/ppo/core_algos.py', {'agg_loss'}, _NAMESPACE)
agg_loss = _NAMESPACE['agg_loss']


class DrGRPOTests(unittest.TestCase):
    def setUp(self):
        self.losses = torch.tensor([[1., 2., 99.], [3., 99., 99.], [4., 5., 6.], [99., 99., 99.]], dtype=torch.float64)
        self.mask = torch.tensor([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.], [0., 0., 0.]], dtype=torch.float64)

    @patch.dict(os.environ, {'MEDTIV_DRGRPO_NORMALIZER': '8192'})
    def test_fixed_normalizer_averages_all_batch_rows(self):
        expected = (self.losses * self.mask).sum() / (4 * 8192)
        torch.testing.assert_close(agg_loss(self.losses, self.mask, 'seq-mean-token-sum-norm'), expected)

    @patch.dict(os.environ, {'MEDTIV_DRGRPO_NORMALIZER': '8192'})
    def test_actor_microbatch_scaling_and_padding_preserve_objective_and_gradient(self):
        expected = (self.losses * self.mask).sum() / (4 * 8192)
        for padding in (0, 7):
            for sizes, dynamic in (((4,), False), ((2, 2), False), ((1, 3), True)):
                with self.subTest(padding=padding, sizes=sizes, dynamic=dynamic):
                    values = torch.nn.functional.pad(self.losses, (0, padding), value=99.).requires_grad_()
                    mask = torch.nn.functional.pad(self.mask, (0, padding))
                    objective = 0
                    offset = 0
                    for size in sizes:
                        # Same scaling as dp_actor.update_policy before backward.
                        scale = size / 4 if dynamic else 1 / len(sizes)
                        objective = objective + agg_loss(values[offset:offset+size], mask[offset:offset+size], 'seq-mean-token-sum-norm') * scale
                        offset += size
                    torch.testing.assert_close(objective, expected)
                    objective.backward()
                    torch.testing.assert_close(values.grad, mask / (4 * 8192))

    def test_opt_in_requires_exact_8192_and_preserves_legacy_loss(self):
        expected = (self.losses * self.mask).sum() / self.mask.shape[-1]
        for setting in (None, '', '1024'):
            with self.subTest(setting=setting), patch.dict(os.environ):
                os.environ.pop('MEDTIV_DRGRPO_NORMALIZER', None)
                if setting is not None:
                    os.environ['MEDTIV_DRGRPO_NORMALIZER'] = setting
                torch.testing.assert_close(agg_loss(self.losses, self.mask, 'seq-mean-token-sum-norm'), expected)

    def test_other_aggregation_modes_unchanged(self):
        for mode in ('token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean'):
            with self.subTest(mode=mode), patch.dict(os.environ):
                os.environ.pop('MEDTIV_DRGRPO_NORMALIZER', None)
                expected = agg_loss(self.losses, self.mask, mode)
                os.environ['MEDTIV_DRGRPO_NORMALIZER'] = '8192'
                torch.testing.assert_close(agg_loss(self.losses, self.mask, mode), expected)


if __name__ == '__main__':
    unittest.main()
