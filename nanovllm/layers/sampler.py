import torch
from torch import nn


class Sampler(nn.Module):
    """按 temperature 从每条序列的词表分布采样一个 token。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Gumbel-max / exponential race 技巧：argmax(p / Exp(1)) 与按 p 分类采样
        # 等价，却只使用逐元素算子和 argmax，适合 torch.compile。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
