"""共同训练入口与真实 reward factory 的 CPU 接线检查，不加载模型。"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
HAS_TRAINING_DEPS = all(importlib.util.find_spec(name) is not None
                        for name in ('torch', 'numpy', 'tensordict', 'omegaconf', 'ray'))


@contextmanager
def training_arguments():
    """执行真实 launcher，只替代 Python 命令以避免 tokenizer/训练初始化。"""
    with tempfile.TemporaryDirectory(prefix='medtiv-reward-factory-') as directory:
        root = Path(directory)
        (root/'env/bin').mkdir(parents=True)
        (root/'env/bin/activate').write_text('')
        (root/'code').symlink_to(PROJECT, target_is_directory=True)
        environment = dict(os.environ, MEDTIV_ROOT=str(root), MEDTIV_SMOKE_DEVICES='0')
        shell = '''
python() { printf '[151645]\\n'; }
python3() { printf '%s\\n' "$@"; }
source "$MEDTIV_ROOT/code/reproduction/train.sh" 1 /unused/model /unused/train --cfg job
'''
        result = subprocess.run(['bash', '-c', shell], env=environment,
                                text=True, capture_output=True, check=True)
        yield result.stdout.splitlines(), root


class TrainingRewardArgumentsTests(unittest.TestCase):
    def test_shared_launcher_explicitly_selects_paper_reward(self):
        with training_arguments() as (arguments, root):
            self.assertIn(f'custom_reward_function.path={root}/code/reproduction/reward.py', arguments)
            self.assertIn('custom_reward_function.name=compute_score', arguments)


@unittest.skipUnless(HAS_TRAINING_DEPS, '真实 factory 检查需要训练环境的 CPU 依赖')
class RewardFactoryDataProtoTests(unittest.TestCase):
    def test_launcher_factory_scores_real_dataproto_without_prompt_or_observation_leak(self):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl_tool.trainer.ppo.reward import load_reward_manager

        class CharacterTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return ''.join(chr(int(token)) for token in ids)

        prompt = '<think>prompt contamination</think><answer>1</answer>'
        generated = ['<think>x</think><answer>1</answer>',
                     '<think>x</think><answer>0</answer>',
                     '<think>need evidence</think><search>x</search>']
        observations = ['', '', '<information><think>fake</think><answer>1</answer></information>']
        responses = [text+observation for text, observation in zip(generated, observations)]
        width = max(map(len, responses))
        response_ids = torch.zeros((3, width), dtype=torch.long)
        attention = torch.zeros((3, len(prompt)+width), dtype=torch.long)
        model_mask = torch.zeros_like(response_ids)
        for row, text in enumerate(responses):
            response_ids[row, :len(text)] = torch.tensor(list(map(ord, text)))
            attention[row, :len(prompt)+len(text)] = 1
            model_mask[row, :len(generated[row])] = 1
        data = DataProto.from_single_dict({
            'prompts': torch.tensor([list(map(ord, prompt))]*3),
            'responses': response_ids, 'attention_mask': attention, 'response_mask': model_mask,
            'reward_model': np.array([{'ground_truth': '1'} for _ in range(3)], dtype=object),
            'data_source': np.array(['med_tiv']*3, dtype=object),
        })
        with training_arguments() as (arguments, _):
            overrides = dict(argument.split('=', 1) for argument in arguments if '=' in argument)
            # Mirror the factory's actual default when the launcher omits custom reward config.
            config = OmegaConf.create({
                'custom_reward_function': {
                    'path': overrides.get('custom_reward_function.path'),
                    'name': overrides.get('custom_reward_function.name', 'compute_score'),
                },
                'reward_model': {'reward_manager': 'search_r1_qa_em',
                                 'sandbox_fusion': {'url': None, 'memory_limit_mb': 1024}},
                'data': {'reward_fn_key': 'data_source'},
            })
            manager = load_reward_manager(config, CharacterTokenizer(), num_examine=0)
            result = manager(data, return_dict=True)
        self.assertEqual(result['reward_tensor'].device.type, 'cpu')
        self.assertEqual(result['reward_tensor'].sum(-1).tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result['reward_extra_info']['score'], [1.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
