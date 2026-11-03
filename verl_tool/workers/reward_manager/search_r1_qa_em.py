"""Search-R1 QA manager wired to the Med-TIV paper reward."""
import torch
import numpy as np
from verl import DataProto
from verl.workers.reward_manager.registry import register
from collections import defaultdict
from reproduction.reward import compute_score as paper_compute_score, decode_generated_response

compute_score = paper_compute_score


@register("search_r1_qa_em")
class SearchR1QAEMRewardManager:
    """
    Reward Manager for Search-R1 style QA tasks with Exact Match scoring.
    """
    name = "search_r1_qa_em"
    
    # fix the error: in reward.py force passing "reward_fn_key" param
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, format_score=0.1, score=1.0, run_id=None, **kwargs) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or paper_compute_score
        self.format_score = format_score
        self.score = score

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards for Search-R1 style responses."""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_length = data_item.batch['prompts'].shape[-1]
            response_ids = data_item.batch['responses']
            response_attention = data_item.batch['attention_mask'][prompt_length:]
            valid_response_length = int(response_attention.sum().item())
            # AgentLoopOutput marks model tokens 1, tool observations/padding 0.
            # Older full-sequence loss_mask uses the same convention.
            generated_mask = data_item.batch.get('response_mask')
            if generated_mask is None and 'loss_mask' in data_item.batch:
                generated_mask = data_item.batch['loss_mask'][prompt_length:]
            sequences_str = decode_generated_response(
                self.tokenizer, response_ids.tolist(), response_attention.tolist(),
                generated_mask.tolist() if generated_mask is not None else None,
            )

            # Get ground truth
            if 'reward_model' in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            else:
                # Fallback to direct ground truth or golden_answers
                ground_truth = data_item.non_tensor_batch.get('ground_truth', 
                              data_item.non_tensor_batch.get('golden_answers', []))

            # Compute score
            score = self.compute_score(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score,
                score=self.score
            )
            if score > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            scores[i] = {"score": score}

            if valid_response_length:
                reward_tensor[i, valid_response_length - 1] = score

            # Print examples for debugging
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"=== Search-R1 QA EM Reward Debug ===")
                print(f"Data source: {data_source}")
                print(f"Score: {score}")
                print(f"Sequence: {sequences_str}")
                print("=" * 50)


        for score_info in scores:
            for key, value in score_info.items():
                reward_extra_info[key].append(value)

        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
