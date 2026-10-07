import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活：把最后一维等分为 gate/up，再计算 SiLU(gate) * up。

    例如输入形状 ``[tokens, 2 * intermediate/tp]``，输出最后一维减半。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
