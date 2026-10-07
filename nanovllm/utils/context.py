from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次 forward 的 attention 元数据。

    模型层签名只传 hidden states；变长序列边界、KV 写入槽位和 block table
    通过这个进程内上下文提供给每一层 Attention。
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """返回当前 forward 共用的上下文。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """在模型执行前一次性安装 attention 参数。

    例如两个 query 长度分别为 3、2 时，``cu_seqlens_q=[0, 3, 5]``；
    FlashAttention 据此从扁平的 5-token tensor 中恢复两条独立序列。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """forward 结束后清空 tensor 引用，避免误用于下一批请求。"""
    global _CONTEXT
    _CONTEXT = Context()
