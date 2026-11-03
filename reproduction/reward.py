"""Med-TIV 论文 Rc × Rf；仅对生成回答评分，不读取提示词。"""
import re

_TAGS = re.compile(r"</?(think|answer)>")
_OBSERVATIONS = re.compile(r"</?(information|tool_response)>")


def strip_observations(text):
    """剥离完整工具观测；截断/错配观测显式失败，禁止使用检索答案。"""
    stack = []
    pieces = []
    cursor = 0
    start = 0
    for match in _OBSERVATIONS.finditer(text):
        if not match.group().startswith("</"):
            if not stack:
                start = match.start()
                pieces.append(text[cursor:start])
            stack.append(match.group(1))
        else:
            if not stack or stack.pop() != match.group(1):
                raise ValueError("工具观测标签未配对")
            if not stack:
                cursor = match.end()
    if stack:
        raise ValueError("工具观测被截断，无法安全评分")
    pieces.append(text[cursor:])
    return "".join(pieces)


def decode_generated_response(tokenizer, response_ids, attention_mask, generated_mask=None):
    """按响应 attention 和生成 mask 取 token，永不接受 prompt IDs。"""
    if len(response_ids) != len(attention_mask):
        raise ValueError("响应 token 与 attention mask 长度不一致")
    if generated_mask is not None and len(generated_mask) != len(response_ids):
        raise ValueError("响应 token 与生成 mask 长度不一致")
    selected = [int(token) for index, token in enumerate(response_ids)
                if attention_mask[index] and (generated_mask is None or generated_mask[index])]
    return strip_observations(tokenizer.decode(selected, skip_special_tokens=True))


def compute_score(solution_str, ground_truth, **_):
    """正确性乘格式分：错误/缺标签为0，正确为1，超过10对answer为0.25。"""
    text = strip_observations(solution_str)
    stack = []
    answers = []
    think_count = 0
    answer_start = None
    for match in _TAGS.finditer(text):
        tag = match.group(1)
        if not match.group().startswith("</"):
            if stack or (tag == "answer" and think_count == 0):
                return 0.0
            stack.append(tag)
            if tag == "answer":
                answer_start = match.end()
        else:
            if not stack or stack.pop() != tag:
                return 0.0
            if tag == "think":
                think_count += 1
            else:
                answers.append(text[answer_start:match.start()].strip())
    if stack or not answers or not think_count:
        return 0.0
    target = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    targets = [target] if isinstance(target, str) else target
    normalize = lambda value: " ".join(value.casefold().split())
    if not answers[-1] or not any(normalize(answers[-1]) == normalize(item) for item in targets):
        return 0.0
    return 0.25 if len(answers) > 10 else 1.0
