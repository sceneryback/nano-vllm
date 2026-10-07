from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求在调度器中的生命周期状态。"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一条生成请求的 token 状态和 KV Cache 映射。

    ``token_ids`` 最初是 prompt，decode 时不断追加生成 token；``block_table``
    则把逻辑块号映射到 GPU 物理 KV block id。
    """
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # 使用显式计数而非 len(token_ids)，因为 TP worker 的 decode 副本只传 last_token。
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

# 向上取整，返回当前 prompt 所需的 block 数量
# 有 @property 后直接像属性一样访问
    @property
    def num_blocks(self):
        # ceil(num_tokens / block_size)。例：257 tokens、block_size=256 -> 2 blocks。
        return (self.num_tokens + self.block_size - 1) // self.block_size

# 最后一个 block 的 token 数量，总量减去前边 n-1 的总 token 数即可
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

# 返回第 i 个 block 的 token id 列表
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

# 添加新 token 到序列的末尾，并更新相关属性
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

# pickle.dumps 序列化时调用，决定保存哪些数据
    def __getstate__(self):
        # decode 阶段 KV cache 已在显存中，恢复时只需要最后一个 token 作为下一步的输入，不需要完整列表
        # prefill 要读取待计算的整段 token；decode 的历史已在 KV cache，只需最后 token。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

# 反序列化
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
