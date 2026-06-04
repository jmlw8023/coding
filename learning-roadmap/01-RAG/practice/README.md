# 01-RAG · Practice Lab

> 配套 [01-RAG/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `rag` conda env（同 [00-基础/practice](../../00-基础-Foundations/practice/README.md) 的 kernel）。

---

## 环境前置

如果你已经跑完 Foundations 章节，jupyter + kernel 已就绪，直接：

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/01-RAG/practice"
jupyter lab
```

新机器请先回 [00-基础/practice/README.md](../../00-基础-Foundations/practice/README.md#环境前置一次性) 装 jupyter + 注册 kernel。

---

## 双模式约定

每个 notebook 顶部都有 `MODE` 开关：

| MODE | 行为 | 何时用 |
|------|------|--------|
| `OFFLINE` | embedding 用 hash-based stub，LLM 用规则字符串拼接 | **默认**，保证不联网也能跑通流程 |
| `ONLINE`  | embedding 调 Ollama `nomic-embed-text`，LLM 调 Ollama `qwen1.5_1.8` | 想看真效果时切；**先在另一终端 `ollama serve`** |

---

## 已知本机依赖坑（已规避）

- ❌ `transformers` 在 `rag` env 因 tokenizers 冲突不可 import → 不用任何依赖 `transformers` 的库
- ❌ `bge-reranker` / `sentence-transformers` 不可用 → 22 号用 **LLM-as-judge** 当 reranker
- ❌ `ragas` / `rank_bm25` / `fastapi` 未装且网络拉不到 → **手撸**这些功能（教学价值反而更高）
- ⚠ Ollama 默认不在跑 → `MODE='ONLINE'` 前需先 `ollama serve`

---

## 学习顺序

按编号顺序走。每个 notebook 设计成：
1. 学习目标 / 预备
2. 阶段拆解（多个代码 cell 渐进 + 讲解）
3. 深入思考 / 反例
4. 自检 ✅
5. 下一步

### Stage 1 · 入门（3 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 16 | [stage1_入门/16_naive_rag_50_lines.ipynb](stage1_入门/16_naive_rag_50_lines.ipynb) | 50 行手撸 naive RAG，OFFLINE/ONLINE 一行切换 | 60 min |
| 17 | [stage1_入门/17_rag_project_walkthrough.ipynb](stage1_入门/17_rag_project_walkthrough.ipynb) | 跑通 [rag_project/](../../../rag_project/) build/query + 改 5 处 config | 60 min |
| 18 | [stage1_入门/18_doc_loaders_and_failure_modes.ipynb](stage1_入门/18_doc_loaders_and_failure_modes.ipynb) | 3 类文档喂同 pipeline 看崩 + loader 选型表 | 60 min |

### Stage 2 · 进阶（3 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 19 | [stage2_进阶/19_chunking_strategies.ipynb](stage2_进阶/19_chunking_strategies.ipynb) | 固定 / 递归 / 语义切块对比，hit@k 表 | 75 min |
| 20 | [stage2_进阶/20_embedding_and_distance.ipynb](stage2_进阶/20_embedding_and_distance.ipynb) | 归一化 / cosine vs L2 等价证明 / 维度三角 | 60 min |
| 21 | [stage2_进阶/21_hybrid_search_handrolled.ipynb](stage2_进阶/21_hybrid_search_handrolled.ipynb) | 手撸 BM25 + dense + RRF 融合 | 90 min |

### Stage 3 · 高级（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 22 | [stage3_高级/22_rerank_hyde_multiquery.ipynb](stage3_高级/22_rerank_hyde_multiquery.ipynb) | LLM-as-judge reranker + HyDE + multi-query | 90 min |
| 23 | [stage3_高级/23_self_corrective_rag.ipynb](stage3_高级/23_self_corrective_rag.ipynb) | Self-RAG / Corrective-RAG 简化版 | 75 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 24 | [stage4_专家/24_eval_and_serving.ipynb](stage4_专家/24_eval_and_serving.ipynb) | 手撸 4 个 RAGAS 风格指标 + 30 题 eval set + RAG 类封装 | 120 min |

> HTTP 服务化 / k8s / 监控 等留给 [06-工程化与部署-Deployment](../../06-工程化与部署-Deployment/)。

---

## 学完之后

回到 [01-RAG/README.md](../README.md) 把 checkbox 打勾，做能力检验（章节末尾 ✅ 列表）。Stage 1–4 全部通过后可进 [02-Agent](../../02-Agent/) 或 [04-模型微调-Finetuning](../../04-模型微调-Finetuning/)。
