# 闸门2 的性能测量记录与纠错

**日期**:2026-10-02 · **平台**:RX 9070 (gfx1201) / Windows 11 / torch 2.13.0+rocm10.0.0
**用途**:闸门2(性能收官)的开工前事实采集。所有数字均有可复现脚本,存于 `tools/`。
**状态**:结论已用于确定实施顺序 `a → b → c`(见文末)。

---

## 一、最终结论(经三轮自我纠错后)

| 优先级 | 杠杆 | 实测依据 | 预期收益 |
|---|---|---|---|
| **a** | 干掉热路径 `.item()` 同步 | 对照实验:28.1 → 36.9 tok/s(**+31%**) | 几分钟改动,立即见效 |
| **b** | triton paged attention kernel | gather 实测 0.8ms/层/步(bs=32)被完全消掉;上游为 0 | 主要收益项 |
| **c** | CUDA Graph 整步捕获 | 图重放 1.3×(含 matmul 的算子流) | 中后期,受 batch 场景约束 |

**关键平台事实**:本机 GPU→CPU 标量同步(`.item()`)约 **3 ms/次**,比正常(~10µs)慢约 300 倍。

---

## 二、纠错记录:三个被推翻的判断

这一节是本文档的主要价值。**三次错误全部源于同一个根因:测量工具自身改变/污染了被测对象。**

### 错误 1:「attention 占 decode 步 43%」——被探针污染

- 第一次用 `probe_decode_profile.py`(cuda events 夹壳 + sync),得 attention 占 30%;
  换 `probe_step_breakdown.py`(events 流时间)得 43%。
- **两个口径自相矛盾**,而真相是:**不带任何探针时 example.py 约 30ms/步(33 tok/s),
  带上探针变 48.8ms/步**——探针自己加了 18ms 以上,两个百分比都是脏数。
- 教训:**先用"无探针基线"锚定,再谈份额**;事件计时本身有成本,尤其在 launch 密集时。

### 错误 2:「96% 时间花在算子派发上」——算错了

- 由 48.8ms/步 减去 GPU 数学地板 2ms,推出"派发占 96%"。
- 实测单算子派发(`probe_op_overhead.py`):**5.4 µs**;280 个算子 ≈ **1.5 ms/步**。
- **派发根本不是瓶颈**,而且 48.8ms 这个分母本身是错的(见错误 1)。

### 错误 3:「gather 成本可接受,图先行」——方向反了

- 曾推荐「先做图安全的纯 torch attention(padded gather),再写 kernel」。
- 实测 `probe_gather_cost.py`:padded gather 在 bs=32/Lk=4096 时 **46.2 ms/层/步**,
  静态缓冲 537 MB——**光 gather 就超过整个 decode 步**。
- 且 **上游为 0**:flash-attn 的 paged kernel 按 block_table 就地读池,不物化连续缓冲。
- 结论:图先行会先造一个必然被拆掉的东西 → **改为 kernel 先行**。

### 真凶的定位过程

`probe_profiler.py`(torch.profiler,低开销)给出:
```
aten::_local_scalar_dense   64.32% CPU   3.026 ms/次   532 次
                          = .item() 的实现      ≈ 20 步 × 28 层
```
对应 `attention_rocm.rocm_with_kvcache` 里的 `Lk = int(cache_seqlens[b])`——**移植时我自己写的**,
每层每序列一次 D2H 同步。上游把 `cache_seqlens` 直接传进 kernel,长度在 GPU 上读,没有这个成本。

`probe_no_item.py` 做唯一变量对照(同进程外其他全同,仅去掉该同步):
**28.1 → 36.9 tok/s**。

---

## 三、探针自身踩的坑(工程教训)

| 坑 | 现象 | 处理 |
|---|---|---|
| cuda events 在 kernel 未完成时查询 | `RuntimeError: Both events must be completed` | 结束时统一收账;该错误本身是 CPU 领先 GPU 的证据 |
| events 夹壳引入同步 | 墙钟从 30ms/步涨到 48.8ms/步 | 改用 torch.profiler |
| `torch.profiler` 自身开销 | 绝对值翻倍(单步 ~28-35ms → profiler 下翻倍) | 只看**相对份额与调用次数**,不看绝对值 |
| 同进程二次初始化进程组 | `trying to initialize the default process group twice!` | 对照实验走独立子进程 |
| 探针校验索引错数组 | 曾把正确 kernel 误报为异常 | 判"平台不可用"前先怀疑探针(见教学区 local-probes) |
| `grep gfx12` 命中 gfx1250 | 差点误报"triton 明写支持 gfx1201" | 精确模式复验 |
| 端口 2333 硬编码 | 并发第二个引擎实例报 `DistNetworkError`(socket bind 失败) | 对照实验必须串行;或改用独立端口 |
| 后台任务无输出 | 长时间 bench 挂后台,stdout 被缓冲,进度不可见 | 关键实验串行前台跑,或加 `-u` 强制无缓冲 |

---

## 四、原始测量数据(留档)

```
probe_attn_breakdown  bs=2 Lk=130 : gather 0.06 + repeatKV 0.02 + matmul 0.57 = 0.65 ms/层
                     bs=32 Lk=1024: gather 0.82 + repeatKV 0.05 + matmul 8.52 = 9.39 ms/层
                     bs=8 Lk=4096 : gather 0.53 + repeatKV 0.13 + matmul 3.01 = 3.67 ms/层
probe_gather_cost     bs=32 Lk=4096: 46.2 ms/步,静态缓冲 536.9 MB,每步流量 15.0 GB
                      bs=8  Lk=4096: 14.5 ms/步,静态缓冲 134.2 MB
probe_op_overhead     单算子派发 5.4 µs;图重放 125.6µs vs 逐个发起 161.6µs ≈ 1.3×
probe_no_item         现状 28.1 tok/s → 去同步 36.9 tok/s
probe_decode_profile  墙钟 7.34s / attention 累计 2.19s(29.9%,脏数)
probe_step_breakdown  墙钟 48.83ms/步 / attention 流时间 21.02ms(43.1%,脏数)
profiler               .item() 3.026ms × 532 次 = 64.3% CPU
```

**无探针基线**:example.py 单条 128 token ≈ **30 ms/步(33 tok/s)**;bench 缩量
(32 条 × 128) = 63.49 tok/s。**以后所有百分比都以无探针基线为分母。**

---

## 五、对闸门2 的影响

1. 实施顺序确定为 **a → b → c**:先拿 a 的即时收益(+31%),再上 b(kernel,消掉 gather),
   最后评估 c(图)——a 同时把「热路径禁止 host 同步」这条约束刻进代码,正是 b 要遵守的纪律。
2. **bench 报数规范**:必须同时报「无探针基线 / 带探针」两组,否则份额不可信。
3. **上游对照的基准**是 README 的 RTX 4070 Laptop 8GB:1434 tok/s(256 条)。
   本机最终数字要与之并列,差距来源需能逐条解释(平台、显存带宽、kernel 成熟度)。

---

## 六、执行结果(同日完成)

三段全部落地,每段都有对照数字:

| 阶段 | 改动 | 结果 |
|---|---|---|
| a | 长度改由 `Context` 的 CPU 侧值提供,不再回读 GPU 标量 | 标量同步 **420 → 0 次**;缩量 bench **63.49 → 86.16 tok/s(+35.7%)** |
| b | triton paged attention,按 `block_table` 就地读池 | 全量 bench **75.2 → 810.9 tok/s**;等价性探针 5 组 ×2 精度全绿 |
| c | decode 步整步图捕获 | bs=1 **4.75×** / bs=8 **3.15×** / bs=32 **1.85×**;eager 与 graph 输出逐 token 一致 |

**图倍率随 batch 递减**是本轮最干净的边界结论:batch 越大,固定开销被摊薄,图的边际收益越小
(bs=1 时 launch 占比高 → 4.75×;bs=32 时 KV 搬运成主项 → 1.85×)。

**全量 bench 与图评估的差距尚未归因**:全量 528.67 tok/s(kernel+图)低于纯 decode 的
1600 tok/s,候选解释有二——① prefill 未做 kernel 化(本轮范围外);② 全量场景的上下文长得多
(decode 成本随上下文线性增长)。`tools/probe_phase_split.py` 是为区分这两者写的,
但**本轮未跑完**(6 轮全量 bench,每轮数分钟,中途终止),所以**不下结论**——
按本仓库纪律,未测量的归因不得写成结论。

### a 段的一个副产品发现

`model_runner.py` 的进程组端口硬编码为 2333。并发跑两个引擎实例时第二个会因端口占用失败:
```
torch.distributed.DistNetworkError: The server socket has failed to listen to any local network address
```
单卡日常使用不受影响(一次一个引擎),但**并行跑对照实验时必须串行**,否则报错。
这个坑在写对照探针时踩到过(见下节"探针踩坑"新增条目)。

