"""最小奖励回归测试；仅依赖 Python 标准库。"""
import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).with_name("reward.py")
_SPEC = importlib.util.spec_from_file_location("medtiv_reward", _PATH) if _PATH.exists() else None
_REWARD = importlib.util.module_from_spec(_SPEC) if _SPEC else None
if _SPEC:
    _SPEC.loader.exec_module(_REWARD)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(_REWARD, "独立论文奖励模块尚未实现")

    def test_correct_without_search(self):
        self.assertEqual(_REWARD.compute_score("<think>reason</think><answer>TRUE</answer>", {"target": "true"}), 1.0)

    def test_wrong_or_missing_final_answer_is_zero(self):
        for response in ("<think>x</think><answer>false</answer>", "<think>true</think>", "<answer>true"):
            with self.subTest(response=response):
                self.assertEqual(_REWARD.compute_score(response, "true"), 0.0)

    def test_malformed_format_is_zero(self):
        for response in ("<answer>true</answer>", "<think>x<answer>true</answer>", "</think><think>x<answer>true</answer>", "<think>x</think><answer>true</answer><answer>"):
            with self.subTest(response=response):
                self.assertEqual(_REWARD.compute_score(response, "true"), 0.0)

    def test_answer_pair_threshold_and_last_answer(self):
        for count, expected in ((10, 1.0), (11, 0.25)):
            response = "<think>x</think>" + "<answer>true</answer>" * count
            self.assertEqual(_REWARD.compute_score(response, "true"), expected)
        self.assertEqual(_REWARD.compute_score("<think>x</think><answer>true</answer><answer>false</answer>", "true"), 0.0)

    def test_no_extra_repetition_penalty(self):
        self.assertEqual(_REWARD.compute_score("<think>" + "However, I should " * 100 + "</think><answer>true</answer>", "true"), 1.0)

    def test_prompt_and_observation_cannot_supply_final_answer(self):
        # Prompt is deliberately not an argument to the response decoder.
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return "".join(chr(token) for token in ids)
        generated = "<think>need evidence</think><search>x</search>"
        observation = "<information><think>x</think><answer>true</answer></information>"
        padded = generated + observation + "padding"
        ids = list(map(ord, padded))
        attention = [1] * (len(generated) + len(observation)) + [0] * 7
        loss_mask = [1] * len(generated) + [0] * (len(observation) + 7)
        text = _REWARD.decode_generated_response(Tokenizer(), ids, attention, loss_mask)
        self.assertEqual(text, generated)
        self.assertEqual(_REWARD.compute_score(text, "true"), 0.0)
        self.assertEqual(_REWARD.decode_generated_response(Tokenizer(), ids, attention), generated)
        with self.assertRaises(ValueError):
            _REWARD.decode_generated_response(Tokenizer(), list(map(ord, "<information>truncated")), [1] * 22)

    def test_mask_length_mismatch_fails(self):
        with self.assertRaises(ValueError):
            _REWARD.decode_generated_response(None, [1, 2], [1], [1, 1])


if __name__ == "__main__":
    unittest.main()
