# Foundations · Practice Lab

> 配套 [00-基础-Foundations/README.md](../README.md) 的 Stage 1 + Stage 2 动手实验代码。
> 全部用 Jupyter notebook 编写，环境 = `rag` conda env（`D:\ProgramData\anaconda3\envs\rag`）。

---

## 环境前置（一次性）

```bash
conda activate rag
pip install jupyter ipykernel matplotlib       # 如已装则跳过
python -m ipykernel install --user --name=rag --display-name "Python (rag)"
```

**验证：**
```bash
conda run -n rag python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.8.0+cu126 True
```

**已知坑（来自本机 memory）：**
- ❌ `import transformers` 在本 env 里会因 `tokenizers` 版本冲突报错。**遇到 transformers 的任务**：用 `tokenizers` 包直接加载，或换独立 env。Notebook 里已避开这个坑。
- ❌ Ollama 默认**没自启动**。需要它的 notebook（08）会提示你先在另一个终端跑 `ollama serve`。

---

## 启动方式

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/00-基础-Foundations/practice"
jupyter lab    # 或 jupyter notebook
```

启动后选 `Python (rag)` kernel。

---

## 学习顺序

按编号顺序走。每个 notebook 设计成：
1. **学习目标** —— 看完能干嘛
2. **预备知识** —— 至少要懂什么
3. **核心代码** —— 多个 cell 渐进，每段有讲解
4. **深入思考** —— 改一改 / 失败案例 / 边界
5. **自检 checklist** —— 这个 notebook 算不算过
6. **下一步** —— 推荐继续读什么

### Stage 1 · 入门（5 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 01 | [stage1_入门/01_python_features.ipynb](stage1_入门/01_python_features.ipynb) | list 推导 / 生成器 / 装饰器 / context manager / async | 45 min |
| 02 | [stage1_入门/02_numpy_basics.ipynb](stage1_入门/02_numpy_basics.ipynb) | 矩阵运算、broadcasting、shape 思维 | 45 min |
| 03 | [stage1_入门/03_git_practice.ipynb](stage1_入门/03_git_practice.ipynb) | branch / merge / rebase / 冲突解决 | 60 min |
| 04 | [stage1_入门/04_vector_math.ipynb](stage1_入门/04_vector_math.ipynb) | 点积 / L2 范数 / 余弦相似度 / softmax 手算 + 验证 | 45 min |
| 05 | [stage1_入门/05_transformer_mental_model.ipynb](stage1_入门/05_transformer_mental_model.ipynb) | 字符串 → token → embedding 全链路 | 60 min |

### Stage 2 · 进阶（4 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 06 | [stage2_进阶/06_mnist_mlp.ipynb](stage2_进阶/06_mnist_mlp.ipynb) | PyTorch 训练循环骨架（MNIST 95%+） | 90 min |
| 07 | [stage2_进阶/07_self_attention.ipynb](stage2_进阶/07_self_attention.ipynb) | 手撸 self-attention，复刻 nn.MultiheadAttention | 90 min |
| 08 | [stage2_进阶/08_qwen_sampling.ipynb](stage2_进阶/08_qwen_sampling.ipynb) | greedy / temperature / top-p 对比（用 Ollama HTTP） | 60 min |
| 09 | [stage2_进阶/09_mini_vecdb.ipynb](stage2_进阶/09_mini_vecdb.ipynb) | 50 行手撸向量库 + 与 chromadb 对比 | 90 min |

### Stage 3 · 高级（4 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 10 | [stage3_高级/10_nano_gpt.ipynb](stage3_高级/10_nano_gpt.ipynb) | 手撸 char-level mini-GPT，唐诗语料训练（MINI 配置 < 1 min，FULL 5–10 min） | 90 min |
| 11 | [stage3_高级/11_tokenizer_compare.ipynb](stage3_高级/11_tokenizer_compare.ipynb) | gpt2 vs cl100k vs o200k 三家 BPE 对比 + 中文成本量化 | 45 min |
| 12 | [stage3_高级/12_bf16_training.ipynb](stage3_高级/12_bf16_training.ipynb) | fp32 / fp16-with-scaler / bf16-autocast 三模式 + warmup + grad clip | 60 min |
| 13 | [stage3_高级/13_attention_deepdive.ipynb](stage3_高级/13_attention_deepdive.ipynb) | `/√d` 方差论证 + 5 种 nan 触发 + RoPE 入门 | 90 min |

### Stage 4 · 专家（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 14 | [stage4_专家/14_llama_block_from_scratch.ipynb](stage4_专家/14_llama_block_from_scratch.ipynb) | RMSNorm + SwiGLU + GQA + RoPE 全部手撸 + 组装 LlamaBlock | 120 min |
| 15 | [stage4_专家/15_kv_cache.ipynb](stage4_专家/15_kv_cache.ipynb) | 手撸 KV-cache，与无 cache 输出 allclose，延迟曲线 + 显存估算 | 90 min |

> Stage 3+4 全部跑完后，**应当能直接看懂 HuggingFace `modeling_llama.py`**，并能估算「某 GPU 能跑多大模型 / 能并发几个用户」。

---

## 学完之后

回到 [00-基础-Foundations/README.md](../README.md) 把对应的 checkbox 打勾，然后做一次 **能力检验**（Stage 1–4 各对应章节末尾的 ✅ 列表）。Stage 3+4 全部通过后，可直接跳到 [01-RAG](../../01-RAG/) 或 [04-模型微调-Finetuning](../../04-模型微调-Finetuning/)。
