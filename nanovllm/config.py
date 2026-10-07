import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """推理引擎配置。

    ``hf_config``、``eos`` 和 ``num_kvcache_blocks`` 是运行时派生字段：前者从
    Hugging Face 模型目录读取，后两者分别由 tokenizer 和显存探测结果回填。
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # Triton 的 KV 写入 kernel 以 256 个元素为基本约束，因此 block_size 必须
        # 是 256 的倍数。默认 256 表示一个物理块容纳 256 个 token 的 K/V。
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 用户上限不能超过模型训练时声明的位置编码上限。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
