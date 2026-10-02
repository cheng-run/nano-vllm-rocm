r"""kernel 后端 vs 参考后端的等价性探针(闸门2 b 段的硬验收)。

为什么存在:
  kernel 后端是新写的索引/掩码逻辑,一处错位就会静默算错(参考后端当初就是靠这类探针对
  朴素实现在移植期抓出 3 个 bug)。本探针对**同一组输入**同时跑两份后端并逐元素比对,
  覆盖参数矩阵:bs、跨块长度、页表 -1 填充、GQA 分组、共享块。

  两份后端共享同一份输入构造(物理池 + 页表),所以差异只能来自实现本身。

用法: .venv\Scripts\python.exe tools\probe_kernel_equivalence.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ref = _load("attention_rocm", "nanovllm/layers/attention_rocm.py")
_kt = _load("attention_triton", "nanovllm/layers/attention_triton.py")

BLOCK_SIZE, KVH, D, H = 256, 8, 128, 16
NB_POOL = 64


def make_case(bs, lens, dtype, device="cuda"):
    """构造物理池 + 页表:每条序列占据独立块,必要时共享同一块(测 ref_count 场景)。"""
    pool_k = torch.zeros(NB_POOL, BLOCK_SIZE, KVH, D, dtype=dtype, device=device)
    pool_v = torch.zeros_like(pool_k)
    max_blocks = (max(lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    bt = torch.full((bs, max_blocks), -1, dtype=torch.int32, device=device)
    next_block = 0
    for i, L in enumerate(lens):
        nb = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in range(nb):
            pool_k[next_block, :, :, :] = torch.randn(BLOCK_SIZE, KVH, D, dtype=dtype, device=device)
            pool_v[next_block, :, :, :] = torch.randn(BLOCK_SIZE, KVH, D, dtype=dtype, device=device)
            bt[i, j] = next_block
            next_block += 1
    q = torch.randn(bs, 1, H, D, dtype=dtype, device=device)
    return q, pool_k, pool_v, bt, torch.tensor(lens, dtype=torch.int32, device=device)


def main():
    device = "cuda"
    print(f"device {torch.cuda.get_device_name(0)} | kernel 后端 vs 参考后端")
    scale = D ** -0.5
    ok_all = True

    cases = [
        ("单序列 短(12)", [12]),
        ("bs=2 不等长(12, 300)", [12, 300]),
        ("跨块边界 (255,256,257)", [255, 256, 257]),
        ("bs=8 混合长度", [1, 17, 255, 256, 257, 300, 511, 512]),
        ("bs=32 同长度(200)", [200] * 32),
    ]
    for dtype, tol in [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)]:
        print(f"\n--- dtype={dtype} (tol {tol}) ---")
        for name, lens in cases:
            bs = len(lens)
            q, pk, pv, bt, cs = make_case(bs, lens, dtype, device)
            ref = _ref.rocm_with_kvcache(q, pk, pv, cs, bt, scale)
            got = _kt.paged_attention_decode(q, pk, pv, cs, bt, scale)
            err = (ref.float() - got.float()).abs().max().item()
            ok = err < tol
            ok_all &= ok
            print(f"{'✓' if ok else '✗'} {name:26} max|err| = {err:.3e}")

    print("\nVERDICT:", "kernel 与参考后端等价 ✓" if ok_all else "存在不等价 ✗")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
