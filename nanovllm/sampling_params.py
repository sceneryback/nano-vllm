from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """单条请求的采样参数；当前实现只支持 temperature sampling。"""
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # temperature 接近 0 等价于 greedy，但本项目的无偏采样公式要求严格为正。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
