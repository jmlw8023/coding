# 01 - RAG · 检索增强生成

> **一句话定位**：把外部知识接进 LLM 上下文，让模型回答它本来不知道的事。
> **时间预算**：4–10 周（与 02/04 章可并行）。
> **适用判断**：能独立设计 chunking + hybrid + rerank + eval 闭环、能为新项目在 1 天内交付能用的 RAG → 进 [05-评估](../05-评估与可观测性-Evaluation/) 或 [06-部署](../06-工程化与部署-Deployment/)。

---

## 学习目标

完成本章后，你应当：
1. 不查文档 30 分钟内写出能跑的朴素 RAG（不依赖 langchain）
2. 给一份失败的 RAG，能 10 分钟内列出 8+ 个排查方向并按优先级排
3. 能为新 RAG 项目在 4 小时内搭好 hybrid + rerank + 4 指标评估
4. 能讲清「什么时候 RAG 不够 / 该用微调 / 该用知识图谱 / 该用 Agent」
5. 能为中型企业写一份 10 页内的 RAG 系统设计文档

---

## 实战状态

✅ **已落地 9 个 notebook**（4 stage 全覆盖，全部 smoke 通过）→ [practice/](practice/)

所有 notebook **offline-first**（hash-based stub），切到 `MODE='ONLINE'` 自动调本机 Ollama。

---

## 全章地图

```
                  ┌──────────────┐
                  │  原始文档     │
                  └──────┬───────┘
                         │ Loader
                         ▼
                  ┌──────────────┐
                  │  文本块       │
                  └──────┬───────┘
                         │ Chunking
                         ▼
                  ┌──────────────┐
                  │  Embedding   │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Vector DB   │
                  └──────┬───────┘
   Query ────► Retrieve ─┤
                         │ Re-rank
                         ▼
                  ┌──────────────┐
                  │  Context     │
                  └──────┬───────┘
                         │ Prompt
                         ▼
                  ┌──────────────┐
                  │   LLM Gen    │
                  └──────┬───────┘
                         ▼
                       Answer
```

每一段都可以**单独优化**，也都可以**单独搞砸**。RAG 的难点不在框架，在「哪一段是当前瓶颈」。

---

## 阶段 1 · 入门：跑通一个能用的 RAG（1–2 周）

### 核心知识点
- 朴素 RAG 的 6 步骤：load → split → embed → store → retrieve → generate
- 嵌入模型与相似度搜索的本质：把文本变成向量，找最近邻
- 向量库的三大件：embedding 列、metadata 列、相似度算法（cosine / L2）
- Prompt 模板的最小结构：`{question} + {retrieved_context} + {instruction}`
- 为什么「能跑通」≠「能用」：召回不准、答案幻觉、引用错位

### 动手任务
- [ ] 跑通本机 [rag_project/](../../rag_project/)：`python -m rag.cli build` → `query "你的问题"`
- [ ] 改 5 处 config（chunk_size、top_k、embedding 模型…），观察答案变化
- [ ] 用 Ollama + `nomic-embed-text` + `qwen1.5_1.8` 自建 50 行内的最小 RAG
- [ ] 用 3 类文档（中文 PDF / DOCX / Markdown）喂同一 pipeline，看哪种崩得最厉害

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 16 | [16_naive_rag_50_lines.ipynb](practice/stage1_入门/16_naive_rag_50_lines.ipynb) | 50 行手撸 RAG + OFFLINE/ONLINE 一行切换 |
| 17 | [17_rag_project_walkthrough.ipynb](practice/stage1_入门/17_rag_project_walkthrough.ipynb) | 走通 rag_project + 改 5 处 config 看效果 |
| 18 | [18_doc_loaders_and_failure_modes.ipynb](practice/stage1_入门/18_doc_loaders_and_failure_modes.ipynb) | pypdfium2/pypdf/docx 选型 + 5 种失败模式 |

### 推荐资源
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) —— 看完后用本机模型重写
- [LlamaIndex Starter](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)
- [Lilian Weng · LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)（同时覆盖 RAG/Agent）
- 本机：[rag-in-action/Readme.md](../../rag-in-action/Readme.md)

### 能力检验
- [ ] 现场画出 RAG 6 步骤流程图，每步说出 1 个常见失败模式
- [ ] 给一份你没见过的 PDF，30 分钟内基于 rag_project 跑通问答
- [ ] 解释「为什么 top_k 不是越大越好」

---

## 阶段 2 · 进阶：每一段都做选型与调参（2–4 周）

### 核心知识点
- **数据加载**：pypdfium2 vs pypdf vs unstructured；扫描件 OCR；表格保留
- **切块**：固定 vs 递归 vs 语义切块；中文标点优先级；overlap 真实作用；父子切块
- **嵌入**：中文（bge-m3 / Conan）vs 英文（text-embedding-3 / nomic）vs 多语言；维度 vs 召回 vs 成本三角
- **向量库**：本地（Chroma / FAISS）vs 服务端（Milvus / Qdrant / pgvector）；归一化 + cosine ≡ 点积
- **检索策略**：稀疏（BM25）+ 稠密（dense）hybrid；metadata filter；MMR 去冗余
- **Prompt 工程**：强制引用、拒答指令、输出格式约束

### 动手任务
- [ ] 在同一份文档上对比 3 种 chunking 策略，做 hit@k 表
- [ ] 实测 embedding 归一化前后排序差异；证明 `‖a-b‖² = 2 - 2·cos(a,b)`
- [ ] 手撸 BM25（80 行内）+ dense + RRF 融合，在「精确关键词查询」上对比单路与混合
- [ ] 实现 hybrid 检索：BM25 + dense + RRF（不依赖 rank_bm25）

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 19 | [19_chunking_strategies.ipynb](practice/stage2_进阶/19_chunking_strategies.ipynb) | 固定 / 递归 / 语义 三种 + hit@3 对比 |
| 20 | [20_embedding_and_distance.ipynb](practice/stage2_进阶/20_embedding_and_distance.ipynb) | 归一化 / cos vs L2 等价证明 / 维度三角 |
| 21 | [21_hybrid_search_handrolled.ipynb](practice/stage2_进阶/21_hybrid_search_handrolled.ipynb) | 手撸 BM25 + dense + RRF |

### 推荐资源
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [BGE 系列](https://huggingface.co/BAAI) —— 中文嵌入代表
- [Anthropic · Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) —— 每个 chunk 加上下文说明，召回 +49%
- 论文：[ColBERT v2 (2022)](https://arxiv.org/abs/2112.01488)、[DPR (2020)](https://arxiv.org/abs/2004.04906)
- 本机：[RAG-Anything/examples](../../RAG-Anything/examples)

### 能力检验
- [ ] 老板说「RAG 召回不准」时，能列出至少 8 个排查方向并按优先级排
- [ ] 能写一份「中文场景 embedding 模型选型报告」（3 候选 + 盲测结论）
- [ ] 能用 200 行内实现 hybrid 检索（BM25 + dense + RRF）

---

## 阶段 3 · 高级：重排、查询重写、Agentic RAG（2–4 周）

### 核心知识点
- **重排**：双塔（bi-encoder）vs 交叉编码（cross-encoder）；`bge-reranker-v2-m3` / Cohere Rerank；LLM-as-reranker（精度高代价大）
- **查询重写**：multi-query / HyDE（让 LLM 写假答案再检索）/ 子问题分解 / Step-Back
- **高级索引**：父子文档 / Auto-merging / Tree index / Summary index / GraphRAG
- **多模态 RAG**：CLIP / Jina-CLIP；表格 Markdown / SQL；版式（LayoutLM / MinerU）
- **Agentic RAG**：Self-RAG / Corrective RAG —— LLM 自评 + 自纠循环

### 动手任务
- [ ] 用 LLM-as-judge 当 reranker（bge-reranker 本机装不上时的兜底）
- [ ] 实现 HyDE + multi-query，在中文技术文档上跑对比表
- [ ] **跑通本机 [RAG-Anything](../../RAG-Anything/examples)** 至少一个 example
- [ ] 实现最小 Self-RAG：LLM 评估「context 是否足够」，不够则再触发一次检索

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 22 | [22_rerank_hyde_multiquery.ipynb](practice/stage3_高级/22_rerank_hyde_multiquery.ipynb) | LLM-as-judge rerank + HyDE + multi-query |
| 23 | [23_self_corrective_rag.ipynb](practice/stage3_高级/23_self_corrective_rag.ipynb) | Self-RAG + Corrective-RAG 自评循环 |

### 推荐资源
- [LangChain · Advanced RAG Cookbook](https://github.com/langchain-ai/rag-from-scratch)
- 论文：[Self-RAG (2023)](https://arxiv.org/abs/2310.11511)、[CRAG (2024)](https://arxiv.org/abs/2401.15884)、[GraphRAG (Microsoft, 2024)](https://arxiv.org/abs/2404.16130)
- [Jerry Liu (LlamaIndex CEO) 博客](https://jerryjliu.medium.com/)
- [Anthropic · Building effective agents](https://www.anthropic.com/research/building-effective-agents)

### 能力检验
- [ ] 给一份失败的 RAG 系统，2 小时内出诊断报告 + 改进方案
- [ ] 口述「双塔 vs 交叉编码」差异 + 适用场景 + 延迟差
- [ ] 用 multi-query + rerank 把中文 hit@5 提升至少 15%

---

## 阶段 4 · 专家：评估、生产、知识图谱、持续学习（持续）

### 核心知识点
- **评估体系**（与 [05](../05-评估与可观测性-Evaluation/) 深度交叉）：
  - 检索段：hit@k / ndcg@k / mrr / context relevance
  - 生成段：faithfulness / answer relevance / 引用准确性
  - 端到端：RAGAS / TruLens / 人工评测
- **生产化要点**：增量索引、删除、版本化、多租户隔离、缓存层、灰度回滚
- **GraphRAG**：实体抽取 → 关系抽取 → 图谱构建 → 社区检测 → Local/Global 查询
- **数据飞轮**：用户反馈 → 难例挖掘 → embedding fine-tune / 重排 fine-tune

### 动手任务
- [ ] 手撸 4 个 RAGAS 风格指标（faithfulness / context_relevance / answer_relevance / context_precision），30 题 eval set 跑 baseline
- [ ] 把 RAG 包成 callable class，建立 baseline.md 追踪每次改动
- [ ] 学习 [ragflow](../../ragflow/) 企业级架构：文档解析流水线 / 任务队列 / 权限模型
- [ ] 用 GraphRAG 在 100 篇文章上构建图谱，做「需要跨文档综合」问题的对照

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 24 | [24_eval_and_serving.ipynb](practice/stage4_专家/24_eval_and_serving.ipynb) | 4 指标手撸 + 30 题 eval + RAG 类封装 + stdlib HTTP |

### 推荐资源
- [RAGAS 文档](https://docs.ragas.io/)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Anthropic · Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex · 生产 RAG 12 痛点](https://www.llamaindex.ai/blog/12-rag-pain-points-and-proposed-solutions)
- 本机：[ragflow/AGENTS.md](../../ragflow/AGENTS.md) 与 [ragflow/CLAUDE.md](../../ragflow/CLAUDE.md)

### 能力检验
- [ ] 能为中型企业写一份「RAG 系统设计文档」（10 页内，含选型 / 风险 / roadmap）
- [ ] 能在 6 小时内把研究原型改造成可灰度上线的服务
- [ ] 能讲清「什么时候 RAG 不够，需要微调 / 知识图谱 / Agent」

---

## 与其它章节的关系

- **上游依赖**：[00-基础 stage 2](../00-基础-Foundations/) 的 attention/embedding 直觉
- **下游被引**：
  - [02-Agent](../02-Agent/) 可把 RAG 作为「retrieval tool」嵌入
  - [04-微调](../04-模型微调-Finetuning/) 可对 embedding 或 reranker 做 fine-tune
  - [07-Capstone 项目 1](../07-综合项目-Capstone/) 是本章的综合演练
- **强耦合**：[05-评估](../05-评估与可观测性-Evaluation/) —— 没评估的 RAG 是赌博，建议 stage 2 就开始接

---

## 反模式

- ❌ **一上来就 GraphRAG**：朴素 RAG 都没调好，先把 baseline 做扎实
- ❌ **盲目升级嵌入**：换更大的模型 ≠ 更好的效果，先评估再换
- ❌ **不看检索内容只看答案**：LLM 编得很顺时容易错觉「系统不错」，实际召回是空的
- ❌ **chunk_size 一把梭**：所有文档同一切块策略，技术手册和小说就该不一样
- ❌ **没有 eval set 就调参**：肉眼看 demo → 调到最后不知道有没有进步
- ❌ **盲信框架**：langchain 升级一次你的代码就坏一次，关键链路自己写

---

## 前沿追踪

- 关键论文 feed：[arXiv cs.IR](https://arxiv.org/list/cs.IR/recent)、[arXiv cs.CL](https://arxiv.org/list/cs.CL/recent)
- 关键工程博客：Anthropic、Pinecone、LlamaIndex、Weaviate、Cohere
- 关键开源仓：[langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)、[microsoft/graphrag](https://github.com/microsoft/graphrag)、[infiniflow/ragflow](https://github.com/infiniflow/ragflow)
