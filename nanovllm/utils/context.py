from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # CPU 侧长度:prepare_* 阶段本来就算出来过(先有 Python 列表,才张量化),
    # 但过去只把张量传下去,逼得参考后端回读 GPU 标量——本机这种 D2H 同步约 3ms/次,
    # decode 每步要付 28 次(闸门2 a 段的根因,对照实测 +31%)。
    # 这里同时保留列表形态,后端按需取用,不再产生同步。
    cache_seqlens_host: list[int] | None = None    # decode:每序列已缓存的 token 数
    cu_seqlens_q_host: list[int] | None = None     # prefill:累积长度(含首元素 0)
    cu_seqlens_k_host: list[int] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, cache_seqlens_host=None, cu_seqlens_q_host=None, cu_seqlens_k_host=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, cache_seqlens_host, cu_seqlens_q_host, cu_seqlens_k_host)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
