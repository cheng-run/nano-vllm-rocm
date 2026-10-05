# nano-vllm-rocm

nano-vllm 的 **AMD ROCm / Windows 适配 fork**——上游 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
依赖 flash-attn,而 flash-attn 在 Windows + ROCm 上没有任何可安装产物。本 fork 用等价实现替换掉它,
并在同一套代码上把性能推回到上游量级。

**目标平台**:AMD RX 9070 (RDNA4 / gfx1201) · Windows 11 · ROCm 10 · torch 2.13.0+rocm10.0.0

## 为什么需要这个 fork

| 上游依赖 | 在 Windows/ROCm 上的状况 | 本 fork 的做法 |
|---|---|---|
| `flash-attn` | PyPI 只有 sdist;ROCm fork 的 release wheel 全是 CUDA/linux;AMD 官方索引无此包 | 两份替代后端(见下) |
| `triton` | AMD 索引只提供 linux wheel | 用社区包 `triton-windows`(import 名同为 `triton`) |
| `nccl` | Windows 版 torch 未编译 NCCL | 单卡回退 `gloo`(多卡仍需 nccl) |
| torch SDPA | 本栈上交替抛 `hipErrorInvalidValue`,且异步污染后续算子 | 禁用,见 `AGENTS.md` |

## 两个 attention 后端

seam 在 `nanovllm/layers/attention.py` 顶部——替换实现不需要改任何调用点。

| 后端 | 做法 | 定位 |
|---|---|---|
| **参考后端**(`attention_rocm.py`) | 按页表 gather 出连续 KV,再分块算 attention | 正确性基准,永久保留 |
| **kernel 后端**(`attention_triton.py`) | triton paged attention,在 kernel 内按 `block_table` 就地读池(即上游 flash-attn 的做法) | 性能路径,只接管 decode |

选后端:`NANOVLLM_ATTN_BACKEND=kernel`(默认 `reference`);开图:`NANOVLLM_EAGER=0`。

## 基准

数字分两组口径,**两组之间不可互相比较**;跨口径并列是本仓库踩过的坑(见文末"口径陷阱")。

### 全量组(与上游 README 同配置:256 序列,输入 100–1024,输出 100–1024,Qwen3-0.6B)

| 配置 | 吞吐 | 说明 |
|---|---|---|
| 参考后端 + eager | **75.2 tok/s** | 闸门 1 的落点 |
| kernel 后端 + eager | **547.0 tok/s** | 消掉 gather + 就地读池 |
| kernel 后端 + 图 | **554.4 tok/s** | 整步捕获重放 |
| 上游 README(RTX 4070 Laptop 8GB) | 1434.1 tok/s | 对照:本机带宽更高(640 vs ~256 GB/s) |

全量组的两个 kernel 数字于 **2026-10-03 复测**(133,966 输出 token,耗时 244.89s / 241.64s);
参考后端的 75.2 为 2026-10-02 的记录值,**本轮未重测**。
同配置下本机 547.0 vs 上游 1434.1——**慢 2.6 倍,而本机带宽还高 2.5 倍**;差距主因尚未归因(见下)。

> **2026-10-03 复测修正**:此前本文档与 `AGENTS.md` 记录的 **kernel+eager = 810.9 tok/s 未能复现**,
> 复测为 **547.0**(−32.5%)。同一会话背靠背两轮中,图配置命中了它自己的历史值(+4.9%),
> 说明并非机器降频所致;且本项目归因表内 bench 配置的 eager 值为 486.7,与 810.9 自相矛盾、
> 反与 547 同量级。**上表已按复测值修正**,810.9 仅在此存档备查。

### 短输出组(bs=32,固定输出 64,仅用于量图的边际收益)

| 配置 | 吞吐 |
|---|---|
| kernel 后端 + 图 | 1600.2 tok/s |

> ⚠️ **1600.2 与上游 1434.1 不可比**:前者是短输出 / 小 batch 场景,后者是全量。
> 引用时不得与全量组数字并列。

**图带来的倍率随 batch 递减**:bs=1 4.75× / bs=8 3.15× / bs=32 1.85×——batch 越大,
固定开销被摊薄,图的边际收益越小。全量组实测仅 **1.01×**(554.4 / 547.0),是同一规律的极端点。

### 全量 bench 数字的归因尝试(四个假设全部被证伪)

`bench.py` 全量(256 序列 / 随机输入 100–1024 / 随机输出 100–1024)测得 **528.7 tok/s**
(2026-10-03 复测 **554.4**),低于短输出组的纯 decode 图评估 1600——**两组口径不同,差距本身不代表问题**;
真正的疑点是全量组内部为何远达不到上游量级,为定位做了四组单变量实验,**四个候选解释没有一个成立**:

| 候选解释 | 实验 | 结果 | 裁决 |
|---|---|---|---|
| prefill 未 kernel 化是主因 | 只切 `NANOVLLM_ATTN_BACKEND`(两相拆分) | prefill 仅 5.55s / 687s = **0.8%** | 否 |
| KV 池超配触发 preempt 重算 | 数 `Scheduler.preempt` 调用次数 | 超配 1.9×、44 次抢占,吞吐 1835 → 1801(几乎不动) | 否 |
| 图在 bench 配置下收益大 | bench 同配置切 eager / graph | 486.7 vs 528.7 = **1.09×**(2026-10-03 复测 547.0 vs 554.4 = **1.01×**,同一结论) | 否 |
| 随机长输出的"尾巴"拖低平均 | 固定输出 64 vs 1024 | 短输出 217 / 长输出 443——**方向相反** | 否 |

**结论:主因仍未知。** 唯一浮现的线索是短输出场景明显更慢(217 vs 443 tok/s),
而 prefill 恰好是短输出场景中相对权重最高、且本项目唯一未实测的一段——
但这是线索不是结论,未经测量不下判断(见 `AGENTS.md` 的纪律)。

**一个口径陷阱**(踩过,值得记):`bench.py` 打印的是**仅输出 token** 的吞吐,
而多个探针算的是**输入 + 输出**;两者不可直接比较——同一实验换口径前后差 16%。

## 安装

```bash
uv sync          # 依赖已锁定在 uv.lock(含 AMD whl-next 索引配置)
```

### ⚠️ 先设 `PYTHONUTF8=1`(中文 Windows 必做)

torch 2.13 的 inductor 在**导入期**用系统默认编码读取 kernel 模板;中文 Windows 上是 GBK,
碰到模板里的非 ASCII 字节就崩——连 `import nanovllm` 都过不去,与本 fork 的代码无关:

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0x94 in position 618: illegal multibyte sequence
```

```powershell
$env:PYTHONUTF8 = "1"      # PowerShell
```
```bash
export PYTHONUTF8=1        # bash / Git Bash
```

权重放在环境变量 `NANOVLLM_MODEL` 指向的目录(默认 `D:\llama.cpp\models\Qwen\Qwen3-0.6B`,
本机约定:按 HF 仓库作者名建一级目录),需要 `config.json` / `tokenizer.json` /
`tokenizer_config.json` / `vocab.json` / `merges.txt` 与 `model.safetensors` 同目录。
换机器只需设这一个变量,不必改代码。

## 如何复现这些数字

所有结论都可由下面几条命令复算。**先做等价性检查,再做性能测量**——这是本项目的纪律:
性能数字只有在正确性有据的前提下才有意义。
(以下命令均需先按"安装"一节设好 `PYTHONUTF8=1`。)

```bash
# ── 正确性(先跑这个)──────────────────────────────
python tools/probe_attention_equivalence.py   # 参考后端 vs 朴素双重循环
python tools/probe_kernel_equivalence.py      # kernel 后端 vs 参考后端(5 组边界 × fp32/bf16)
python tools/probe_zero_sync.py               # decode 步内 D2H 标量同步 = 0
python tools/probe_graph_equivalence.py       # eager 与 graph 输出逐 token 一致
python example.py                             # 端到端:输出应连贯且推理正确

# ── 性能(无探针基线,不做任何插桩)────────────────
NANOVLLM_EAGER=1 python bench.py                                # 参考后端 + eager
NANOVLLM_ATTN_BACKEND=kernel NANOVLLM_EAGER=1 python bench.py   # kernel 后端 + eager
NANOVLLM_ATTN_BACKEND=kernel python bench.py                    # kernel 后端 + 图  ← 主结果
python tools/probe_graph_eval.py                                # 图倍率随 batch 的变化
```

> **参考后端只能测 eager**:它在 CUDA Graph 捕获期会回退到 `cache_seqlens.tolist()`,
> 而捕获期间禁止 CPU↔CUDA 拷贝,直接报
> `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture`。
> 要开图必须用 kernel 后端——这正是 kernel 后端"图安全"的含义。

**测量规范(踩过坑,写进 `AGENTS.md` 了)**:性能份额必须用无探针基线。
cuda events 夹壳这类探针会给每步加 ~18ms,足以让"attention 占 43%"这类结论完全失真
(见 `notes/2026-10-02-performance-measurements.md`)。

**并行跑对照实验**:`model_runner.py` 的进程组监听端口默认 2333(沿用上游值),同机两个引擎实例
会用同一端口而撞车。设 `NANOVLLM_MASTER_PORT` 给其中一个换端口即可并行——本项目早期的对照
实验全部靠串行绕开,现在不必了。

## 验证脚本(`tools/`)

每个结论都有可复现的脚本,不删——证据价值。

| 脚本 | 断言什么 |
|---|---|
| `probe_attention_equivalence.py` | 参考后端 vs 朴素双重循环(右下 causal / GQA / `-1` 填充) |
| `probe_kernel_equivalence.py` | **kernel 后端 vs 参考后端**(bs、跨块、混合长度、GQA) |
| `probe_zero_sync.py` | decode 步内 D2H 标量同步次数为 0(防回归) |
| `probe_graph_equivalence.py` | eager 与 graph 路径输出一致 |
| `probe_graph_eval.py` | 图在哪些 batch 区间划算 |
| `probe_phase_split.py` | 差距归因:两后端单变量对照,分离 prefill / decode 份额 |
| `probe_preemption.py` | 差距归因:数 `Scheduler.preempt` 次数,验证 KV 池超配假说 |
| `probe_sdpa_stability.py` | 为何禁用 SDPA(留档) |
| `probe_decode_profile.py` · `probe_step_breakdown.py` | 早期耗时画像(**探针有污染,结论已废**,留作反面教材) |
| `probe_gather_cost.py` · `probe_attn_breakdown.py` · `probe_op_overhead.py` · `probe_no_item.py` · `probe_profiler.py` | 定位瓶颈的中间探针 |

## 文档

- `CONTEXT.md` — 本 fork 的术语表(后端 / seam / 图安全 / 无探针基线…)
- `notes/2026-10-02-performance-measurements.md` — 测量记录、三轮自我纠错与 2026-10-03 独立复测
- `AGENTS.md` — 工程纪律(测量规范、禁止项)
