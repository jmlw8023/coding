# 00 - 基础 Foundations

> **一句话定位**：补齐做 RAG / Agent / 微调之前**必须**先有的工程与理论底子。
> **时间预算**：1–4 周（依底子而定）。
> **适用判断**：能独立写出一个 200 行的 PyTorch 训练脚本、能讲清 Attention 公式各项含义 → 直接跳到 01。

---

## 学习目标

完成本章后，你应当：
1. 能独立创建/切换/导出 Python 环境，且解释 conda / pip / uv 的差异
2. 能用 NumPy / PyTorch 实现简单矩阵运算与梯度反传，理解 `tensor.requires_grad`、`backward()` 在干什么
3. 能讲清 LLM 的「输入 → tokenizer → embedding → Transformer block × N → logits → 采样」全链路
4. 能写出至少 3 种不同结构的有效 prompt（zero-shot / few-shot / CoT），并解释为什么这么写
5. 能用 `git` 完成 branch / rebase / cherry-pick / 解决冲突
6. **专家级**：能从零搭出 Llama-style block（RMSNorm + SwiGLU + GQA + RoPE）+ KV-cache

---

## 实战状态

✅ **已落地 15 个 notebook**（4 stage 全覆盖，全部 smoke 通过）→ [practice/](practice/)

各 stage 的代码、讲解、自检都在 practice 里，**强烈建议从 notebook 开始走**，本 README 只作为目录索引与理论补充。

---

## 全章地图

```
       Python · Git · 数据结构 · 命令行
                    │
                    ▼
       NumPy · PyTorch · 训练循环
                    │
                    ▼
       Attention · Tokenizer · LLM 心智模型
                    │
                    ▼
       Pre-training 直觉 · Scaling Law
                    │
                    ▼
       Llama Block · KV-cache · 推理优化
```

---

## 阶段 1 · 入门（必修）

### 核心知识点
- **Python 工程基础**：虚拟环境、`requirements.txt` / `pyproject.toml`、`__init__.py` 与 `__main__.py`、相对/绝对导入
- **命令行 & Shell**：`grep / find / xargs`、管道、I/O 重定向。Windows 下用 Git Bash 即可
- **Git 工作流**：`add / commit / push / pull / branch / merge`，`.gitignore` 的常见坑
- **数学最小集**：向量、矩阵乘法、点积/余弦相似度、概率与条件概率、softmax、交叉熵
- **LLM 最小心智模型**：什么是 token、embedding、上下文窗口、「概率续写」

### 动手任务（与 notebook 直接对应）
- [ ] 写 5 个 Python 特性 demo（list 推导 / 生成器 / 装饰器 / `with` / `asyncio.gather`）
- [ ] NumPy 基础：shape / broadcasting / 矩阵乘法 / softmax
- [ ] 在沙箱里做完整 git 工作流：init → commit → branch → merge → rebase → 解冲突
- [ ] 手算 + NumPy 验证：点积 / L2 范数 / 余弦相似度 / softmax / 交叉熵
- [ ] 用真实 tokenizer（tiktoken）走通 string → token → embedding → next token 全链路

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 01 | [01_python_features.ipynb](practice/stage1_入门/01_python_features.ipynb) | list/gen/decorator/with/asyncio |
| 02 | [02_numpy_basics.ipynb](practice/stage1_入门/02_numpy_basics.ipynb) | shape/broadcasting/矩阵乘/softmax |
| 03 | [03_git_practice.ipynb](practice/stage1_入门/03_git_practice.ipynb) | branch/merge/rebase/冲突解决 |
| 04 | [04_vector_math.ipynb](practice/stage1_入门/04_vector_math.ipynb) | 点积/L2/cosine/softmax/CE 手算 |
| 05 | [05_transformer_mental_model.ipynb](practice/stage1_入门/05_transformer_mental_model.ipynb) | string→token→embedding→logits 全链路 |

### 推荐资源
- 《Python Cookbook》（Beazley）—— 选读
- [CS50P (Harvard)](https://cs50.harvard.edu/python/) —— 系统补 Python
- [Pro Git 中文版](https://git-scm.com/book/zh/v2) —— 前 3 章足够
- [3Blue1Brown · 神经网络系列](https://www.3blue1brown.com/topics/neural-networks) —— 数学直觉
- [Andrej Karpathy · Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) —— 从加法到 GPT 手撸

### 能力检验
- [ ] 现场创建一个新 conda env，安装 `numpy + torch`，跑通一段矩阵乘法
- [ ] 现场画图：把 "I love NLP" 从字符串到 token id 到 embedding 的过程画出来
- [ ] 现场口算：softmax([1, 2, 3]) 的近似结果（≈ [0.09, 0.24, 0.67]）
- [ ] 现场 git：模拟一个冲突并解决
- [ ] 解释「为什么 LLM 不能做精确加法」

---

## 阶段 2 · 进阶（强烈建议）

### 核心知识点
- **NumPy / PyTorch 进阶**：broadcasting 规则、view vs reshape、`einsum`、autograd 计算图
- **训练循环 5 步骨架**：`zero_grad → forward → loss → backward → step`
- **常见网络层**：Linear、LayerNorm、Dropout、Embedding、MultiHeadAttention
- **采样策略**：greedy / top-k / top-p / temperature 的实际效果差别
- **向量库底层**：50 行 brute-force 实现 + chromadb 对照

### 动手任务
- [ ] PyTorch 训 MNIST MLP 到 95%+ 准确率，全程 100 行内
- [ ] 手撸 multi-head self-attention，结果与 `nn.MultiheadAttention` **逐元素对齐**
- [ ] 用 Ollama HTTP 跑 Qwen，对比 greedy / temperature / top-p 的稳定性
- [ ] 50 行手撸向量库，与 chromadb 在 10k 向量上对比

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 06 | [06_mnist_mlp.ipynb](practice/stage2_进阶/06_mnist_mlp.ipynb) | PyTorch 训练 5 步骨架（实测 MNIST 97.87%） |
| 07 | [07_self_attention.ipynb](practice/stage2_进阶/07_self_attention.ipynb) | 手撸 MHA vs nn.MultiheadAttention（差 0） |
| 08 | [08_qwen_sampling.ipynb](practice/stage2_进阶/08_qwen_sampling.ipynb) | greedy/temp/top-p（OFFLINE 数学 + ONLINE Ollama） |
| 09 | [09_mini_vecdb.ipynb](practice/stage2_进阶/09_mini_vecdb.ipynb) | 50 行向量库 + chromadb 对比 |

### 推荐资源
- [PyTorch 60-min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [HuggingFace Course (free)](https://huggingface.co/learn/llm-course/chapter1/1) —— 前 4 章
- [Karpathy · makemore 系列](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [einops 教程](https://einops.rocks/)

### 能力检验
- [ ] 不查文档，5 分钟内写出 PyTorch 训练循环骨架
- [ ] 解释 `tensor.detach()` / `torch.no_grad()` / `tensor.requires_grad_(False)` 的区别
- [ ] 看到 `(B, H, L, L)` 张量，立刻说出每一维是什么
- [ ] 给一份 MNIST acc 卡在 80% 的脚本，5 分钟内提出 3 个排查方向

---

## 阶段 3 · 高级（看方向选学）

### 核心知识点
- **Attention 推导**：从 `QKᵀ/√d` 到带 mask 的 causal attention，能在白板上手推
- **位置编码**：绝对位置 vs RoPE vs ALiBi，长上下文的本质问题
- **训练稳定性**：LayerNorm vs RMSNorm、pre-norm vs post-norm、grad clipping、warmup
- **混合精度 & 显存**：fp16 / bf16 / fp8、激活检查点、梯度累积
- **分词器**：BPE / WordPiece / SentencePiece、不同家族的中文差异

### 动手任务
- [ ] **手撸 nano-GPT**：在唐诗语料上训练，看 loss 收敛
- [ ] 对比 `gpt2` / `cl100k` / `o200k` 在中英混合文本上的 token 数差异
- [ ] 同模型分别用 `fp32` / `fp16+GradScaler` / `bf16+autocast` 训 MLP，对比 loss / 显存 / 时间
- [ ] 数值演示「为什么 `/√d_k`」+ 5 种 nan 触发方式 + RoPE 旋转不改向量长度

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 10 | [10_nano_gpt.ipynb](practice/stage3_高级/10_nano_gpt.ipynb) | 手撸 char-level GPT，唐诗训练 3 秒收敛 |
| 11 | [11_tokenizer_compare.ipynb](practice/stage3_高级/11_tokenizer_compare.ipynb) | 三家 BPE + 中文成本量化 |
| 12 | [12_bf16_training.ipynb](practice/stage3_高级/12_bf16_training.ipynb) | fp32/fp16/bf16 实测 + grad clip + warmup |
| 13 | [13_attention_deepdive.ipynb](practice/stage3_高级/13_attention_deepdive.ipynb) | `/√d` 方差证明 + nan + RoPE 入门 |

### 推荐资源
- [Karpathy · nanoGPT 源码](https://github.com/karpathy/nanoGPT) —— **学完整个项目，不只是看**
- [Lilian Weng · Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GPT-3](https://arxiv.org/abs/2005.14165)、RoPE、FlashAttention

### 能力检验
- [ ] 白板手推 causal self-attention 完整公式（含 scaling、mask、softmax）
- [ ] 现场调试：给一个 loss = nan 的训练脚本，5 步内定位原因
- [ ] 解释「为什么 1B 模型在长文本上常常忘记开头」（位置编码 + attention 复杂度两角度）

---

## 阶段 4 · 专家（按需）

### 核心知识点
- **从零实现 Llama Block**：RMSNorm + SwiGLU + GQA + RoPE
- **KV-cache**：原理、实现、空间换时间公式
- **推理优化**：speculative decoding、quantization、batching
- **缩放定律**：Chinchilla 法则，参数量 vs token 数 vs 计算量

### 动手任务
- [ ] 从零手撸 Llama-style block（4 子模块 + 组装），全部 sanity test 通过
- [ ] 手撸 KV-cache，验证「有 vs 无」输出 `torch.allclose`，画延迟曲线
- [ ] 精读论文：[Chinchilla (2022)](https://arxiv.org/abs/2203.15556)、[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 14 | [14_llama_block_from_scratch.ipynb](practice/stage4_专家/14_llama_block_from_scratch.ipynb) | RMSNorm + SwiGLU + GQA + RoPE 全手撸 + 组装 |
| 15 | [15_kv_cache.ipynb](practice/stage4_专家/15_kv_cache.ipynb) | 两版本 allclose 验证 + 延迟曲线 + 显存估算 |

### 推荐资源
- [Llama 3 论文](https://arxiv.org/abs/2407.21783) —— 当前最好的「现代 LLM 实战手册」
- [Sebastian Raschka · LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

### 能力检验
- [ ] 在 1 小时内，从空文件写出能跑的 mini-Llama block，loss 能下降
- [ ] 给一段任意 LLM 推理优化博客，能立刻指出哪些是 marketing 哪些是真创新
- [ ] 能给同事讲一节 1 小时「Transformer 架构演进」分享课

---

## 与其它章节的关系

- **下游被引**：
  - [01-RAG](../01-RAG/) 大量复用 stage 2 的 attention/embedding 直觉
  - [02-Agent](../02-Agent/) 复用 stage 1 的 asyncio + tool use 概念
  - [04-模型微调](../04-模型微调-Finetuning/) 必须先理解 stage 3/4 的训练稳定性 + 精度问题
- **上游依赖**：无，本章是路线起点
- **并行可学**：[05-评估](../05-评估与可观测性-Evaluation/) 的 metric 部分可以提前看

---

## 反模式

- ❌ **跳着学**：以为「Python 我会，不用看 01」—— 但 asyncio.gather / 装饰器堆叠在生产 LLM 代码里频繁出现
- ❌ **理论先行**：看完三本机器学习书都没跑过一次 `.backward()` —— 直接进 06 号 notebook
- ❌ **只跑不思考**：notebook 跑通就关掉，不做能力检验 → 隔天忘光
- ❌ **跳过手算**：04 号 notebook 让你手算 softmax 是因为「以后看公式时你能秒懂」，不是因为算盘
- ❌ **轻视 git**：80% 的「代码丢了」事故都是 git 没用熟

---

## 前沿追踪

- 关键 newsletter：[The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/)、[Import AI](https://importai.substack.com/)
- 关键人物：Andrej Karpathy、Lilian Weng、Sebastian Raschka、Jeremy Howard
- 关键会议：NeurIPS / ICLR / ACL（**只读你看得懂的**，不强行卷）
- 关键 repo：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)、[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)、[lucidrains/](https://github.com/lucidrains)
