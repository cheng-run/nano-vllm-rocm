# nano-vllm-rocm

本 fork 把 nano-vllm 移植到 AMD RX 9070 (gfx1201) / Windows / ROCm。这里的词分两层:
推理引擎的通用词汇(block、prefill、decode、`num_cached_tokens`…)沿用上游与 vLLM 生态的
通行含义,**不在此重复定义**;本文件只收录移植过程中新增、或在上游没有对应的概念。

## Language

### 移植结构

**后端 (backend)**:
位于 flash-attn 兼容 seam 之后的一份 attention 实现。seam 是 `attention.py` 顶部的
`flash_attn_varlen_func` / `flash_attn_with_kvcache` 两个名字,换实现不改调用点。
目前有两份:参考后端与 kernel 后端。
_Avoid_: 实现、引擎、driver

**参考后端 (reference backend)**:
`layers/attention_rocm.py` 里用纯 torch 算子手写的那份后端:按页表 gather 出连续 KV,
再分块算 attention(右下 causal、GQA、`-1` 填充)。它已被朴素双重循环全绿验证,是
kernel 后端的**正确性基准**,永久保留,不因性能落后而删除。

**kernel 后端 (kernel backend)**:
`layers/attention_triton.py` 里自写的 triton paged attention 后端,**只接管 decode**。
按 `block_table` 在 kernel 内就地读 KV——即上游 flash-attn 的做法,消掉参考后端的 gather 搬运。
长度与页表都在 kernel 内从设备张量读取,因此天然**图安全**,是开启 CUDA Graph 的前提。
_Avoid_: 加速版、优化版

**seam**:
`attention.py` 顶部那两个名字所构成的替换点。判断某处改动是否"换后端"的标准:
是否需要改 seam 之外的调用点。

### 性能约束

**图安全 (graph-safe)**:
一段代码可以在 CUDA Graph 捕获期内执行的性质:无 host 同步(`.item()`、
`int(tensor)`、`.tolist()`)、形状静态、无 CPU 分支依赖 GPU 值。
本仓库的硬约束——热路径上的任何新代码都必须图安全。
_Avoid_: 可捕获、异步安全

**图的边际收益递减**:
图把一个 decode 步从"几百次算子提交"压成"一次提交",省下的是**固定开销**。
batch 越大,固定开销被摊薄,KV 搬运成为主项——本机实测 bs=1 4.75× →
bs=8 3.15× → bs=32 1.85×。因此"开图更快"必须带 batch 区间说,不能笼统宣称。

**热路径 (hot path)**:
每个 decode 步、每层都要执行的代码。当前热路径的图安全违规点已经清除
(参见测量文档的 `.item()` 纠错)。
_Avoid_: 主循环、关键路径

**无探针基线 (probe-free baseline)**:
不做任何插桩时测得的端到端耗时。**所有性能份额必须以它为分母**——
本项目的第一次份额测量就因探针自身引入 18ms/步而得出错误结论。

### 交付节奏

**闸门 1 (gate 1)**:
"能跑"的验收:example.py 与 bench.py 在目标平台上出结果。已通过(2026-09-30)。

**闸门 2 (gate 2)**:
"跑得像样"的验收:一个性能增量周期,顺序 `a → b → c`
(a 清除热路径同步 / b 写 kernel 后端 / c 评估图捕获)。
对外是一个故事("把移植引擎从 63 tok/s 提到 X"),工程上分三步。
_Avoid_: 优化阶段、第二阶段

**降级路径 (degrade path)**:
每个 checkpoint 未过时的退路:kernel 卡死 → 停在参考后端+图外的数字收尾;
图的坑 → 退 eager。闸门哲学是"损失封顶",降级不是失败。
