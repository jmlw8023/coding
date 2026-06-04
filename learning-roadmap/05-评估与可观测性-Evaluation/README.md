# 05 - 评估与可观测性 Evaluation & Observability

> **一句话定位**：把「肉眼看 demo」升级到「有数据支持的迭代」。每次改动你都能说出「提升 / 退步多少、在哪类样本上」。
> **时间预算**：2–6 周（贯穿其它所有章节，**不要等到最后才学**）。
> **适用判断**：团队评估系统能在 1 天告诉你「上周新版本好坏 + 差在哪类 case」、没人会跳过 eval 直接上线 → 本章已落地。

---

## 学习目标

完成本章后，你应当：
1. 给一个 RAG / Agent 项目，4 小时内搭起 LLM-as-judge + 自定义指标的评估流水线
2. 不查文档说出 RAGAS 4 个核心指标各衡量什么 + 怎么解读
3. 能设计严谨的 A/B 实验（含样本量计算）
4. 能为新项目接好完整可观测（trace + metric + alert）
5. 能讲清「judge 偏见 3 类 + 各自缓解手段」

---

## 实战状态

⏳ **notebook 待建**（路线已写）。已经在 [01-RAG 24 号 notebook](../01-RAG/practice/stage4_专家/24_eval_and_serving.ipynb) 实现了 4 个 RAGAS 风格指标手撸版本 + 30 题 eval set，可作起点。

---

## 全章地图

```
┌─────────────────────────────────────────────────┐
│   1. 离线评估（开发期）                            │
│      ─ 单元 eval：每个组件单独跑                    │
│      ─ 端到端 eval：用户视角问答                    │
│      ─ 回归测试：每次改动跑一遍                     │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│   2. 在线评估（生产期）                            │
│      ─ 用户反馈（点赞/吐槽）                       │
│      ─ 隐式信号（停留 / 复制 / 复问）               │
│      ─ A/B 实验                                  │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│   3. 可观测性（永远在线）                          │
│      ─ trace（每次请求的完整链路）                  │
│      ─ metric（延迟 / 成本 / 错误率）              │
│      ─ alert（异常自动告警）                       │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│   4. 数据飞轮                                    │
│      生产 trace → 失败 case → 加入 eval set →     │
│      调整 prompt/RAG/模型 → 上线 → 再循环          │
└─────────────────────────────────────────────────┘
```

---

## 阶段 1 · 入门：基础指标与第一份 eval set（半–1 周）

### 核心知识点
- **基础 NLP 指标**：EM / F1 / BLEU / ROUGE / BERTScore —— 对「能背标准答案」的任务有用，对开放问答**不够**
- **检索段专用**：Recall@k / Precision@k / MRR / NDCG@k / Hit Rate
- **eval set 最小要件**：至少 30 题；覆盖易/中/难 + 常见/边界 + 正面/反面；每题标注答案 + 相关文档 id + 难度 + 类型
- **存放**：JSON / JSONL 跟代码一起 git

### 动手任务
- [ ] 为 [rag_project/](../../rag_project/) 造 30 题 eval set（5 事实 + 5 推理 + 5 多文档 + 5 应拒答 + 10 日常），存为 `rag_project/eval/eval_v1.jsonl`
- [ ] 写 < 100 行评估脚本：读 eval set → 调 rag_project 查询接口 → 算 EM / 包含匹配 → 输出 markdown 表
- [ ] 跑 baseline，记下数字（如 EM=0.4, Recall@5=0.7）
- [ ] 改一个参数（如 chunk_size），再跑一次对比

### 配套 notebook
⏳ 待建（计划：基础指标 / eval set 设计 / baseline 表）

### 推荐资源
- [HF Evaluate 库](https://huggingface.co/docs/evaluate)
- [Pinecone Vector Search Eval](https://www.pinecone.io/learn/series/rag/eval-rag/)
- 《Speech and Language Processing》(Jurafsky & Martin) 评估章

### 能力检验
- [ ] 1 小时内为新 RAG 项目写 20 题 eval set
- [ ] 不查文档说出 Recall@k vs Precision@k 差异
- [ ] 解释「为什么 BLEU 不适合评 RAG 答案」

---

## 阶段 2 · 进阶：LLM-as-judge、RAGAS、组件评估（1–2 周）

### 核心知识点
- **LLM-as-judge**：用更强模型评估目标模型；评分维度可定制（相关性 / 忠实度 / 完整度 / 流畅度）
  - **必备技巧**：给 rubric / CoT 先论据后打分 / 多次评同题求均值降方差 / pairwise 比较更稳
- **RAGAS 框架**：
  - 检索段：`context_precision` / `context_recall` / `context_relevance`
  - 生成段：`faithfulness` / `answer_relevance` / `answer_correctness`
  - 部分指标不需要 ground truth
- **组件分层评估**：**不要只评端到端**；单独评 embedding（MRR） / 检索（Recall@k） / 重排（NDCG@k） / 生成（忠实度）
- **Agent 评估特殊性**：不只看最终答案，还要看 trajectory；LLM-as-judge 做轨迹 critique

### 动手任务
- [ ] **接入 RAGAS**（或手撸版本，已在 [01-RAG 24 号](../01-RAG/practice/stage4_专家/24_eval_and_serving.ipynb) 实现）：5 指标跑 rag_project，对比阶段 1 的 EM
- [ ] **写自定义 LLM-judge**：rubric 相关性 1-5、忠实度 1-5、引用强制；JSON schema 强制输出；与 self-consistency（同题问 3 次取众数）对比稳定性
- [ ] **组件分层实验**：50 条 (query, relevant_doc_ids) 标注，单独评 embedding / 检索 / re-rank，看到底是哪一段拖累端到端
- [ ] **建立 baseline 表**：所有指标列一份 baseline `eval/baseline.md`，每次改动加一列

### 配套 notebook
⏳ 待建（计划：手撸 4 指标已在 24 号，本章重点 LLM-judge rubric / 分层 / pairwise）

### 推荐资源
- [RAGAS 文档](https://docs.ragas.io/)
- [LangChain · LLM-as-judge](https://docs.smith.langchain.com/evaluation/concepts)
- 论文：[Judging LLM-as-a-Judge (2024)](https://arxiv.org/abs/2306.05685)、[G-Eval (2023)](https://arxiv.org/abs/2303.16634)
- [Eugene Yan · Evals](https://eugeneyan.com/writing/evals/)

### 能力检验
- [ ] 4 小时内为新 RAG 系统搭起 RAGAS 5 指标 + 自定义 judge
- [ ] 讲清 LLM-as-judge 3 类偏见 + 3 种缓解手段
- [ ] 根据 baseline 表告诉团队「下次该优化哪一段」

---

## 阶段 3 · 高级：人工评测、A/B 测试、可观测平台（1–2 周）

### 核心知识点
- **人工评测协议**：何时必须做（上线 sign-off / 模型大版本变更 / RLHF 数据采集）；评分维度清晰互斥 + 一致性检验（Cohen's kappa）+ 盲评 + 随机抽样；工具：Argilla / Label Studio / Streamlit
- **A/B 测试**：流量分桶；北极星指标 + 护栏指标；显著性（样本量 + 置信区间）；**切勿每天换实验、看小数点变化下结论**
- **可观测平台**：
  - **Langfuse**（开源 + SaaS，专为 LLM trace，可本地部署）
  - LangSmith（LangChain 官方）
  - Phoenix / Arize（强 embedding 可视化 + drift）
  - W&B / MLflow（通用）
  - OpenTelemetry（底层）
- **每次请求 trace**：输入 prompt + 检索 context + tools / LLM 输出 + 模型 + 各阶段延迟 / token + 用户 ID / 会话 ID / trace ID / 用户反馈
- **关键运行指标**：性能（p50/p95/p99 / QPS / 错误率）/ 质量（拒答率 / token-per-请求 / 用户满意度）/ 成本（每请求美元 / 按租户拆）/ 健康（截断率 / tool 失败率 / 超时率）

### 动手任务
- [ ] **接 Langfuse**（本地 Docker 一键启动）：rag_project 每次查询上传 trace；dashboard 看 p95 / token 分布 / 错误样本；加 thumbs up/down
- [ ] **设计 A/B 实验**：对比「加 re-rank vs 不加」，定义指标 + 算样本量 + 跑实验 + 出报告
- [ ] **搭简易人工评测平台**：Streamlit 写一个页面，加载随机 10 条 trace，让你或同事打 1-5 分，存 SQLite，算 inter-rater agreement
- [ ] **失败 case 自动捞**：Langfuse 设规则「差评 / latency > 5s / tool 失败」→ 自动归档「待优化」队列

### 配套 notebook
⏳ 待建（计划：Langfuse 接入 / A/B 样本量计算 / Streamlit 人工评测）

### 推荐资源
- [Langfuse 文档](https://langfuse.com/docs)
- [Phoenix (Arize)](https://docs.arize.com/phoenix)
- [Argilla 文档](https://docs.argilla.io/)
- [Eugene Yan · Evaluating LLMs is a minefield](https://eugeneyan.com/writing/llm-evaluators/)
- 《Trustworthy Online Controlled Experiments》—— A/B 经典

### 能力检验
- [ ] 为生产 RAG 接好完整可观测（trace + metric + alert）
- [ ] 为一个改动设计严谨 A/B 方案（含样本量计算）
- [ ] 跑一次人工评测，给出 inter-rater agreement，并解释为什么是这个数

---

## 阶段 4 · 专家：数据飞轮、持续优化、跨团队治理（持续）

### 核心知识点
- **数据飞轮**：生产 trace → 失败标注 → 加入 eval → 训练数据 → 模型/prompt 优化 → 上线 → 再循环；**飞轮速度决定团队进步速度**；关键是「捞失败 case」要自动化
- **评估治理**：每条 eval 题有 owner + last_reviewed；定期清过时题；任何引发事故的 case **必须**加入 eval；prompt 变更必须跑全量 regression
- **评估自动化**：CI 集成（PR 触发 eval、结果贴 PR 评论）；阈值告警（退化 > X% 自动 block）；周报（自动生成各模块指标变化）
- **评估的元评估**：你的 judge 准不准？周期性人工抽检 + Pearson 相关性；eval set 覆盖度够不够？看生产 trace vs eval 分布差距

### 动手任务
- [ ] **失败案例自动归档**：脚本每天从 Langfuse 拉「差评 + 失败」样本，自动按主题聚类，出周报
- [ ] **CI 接入 eval**：PR 触发 50 题 regression，结果作为 PR comment，关键指标退化 > 5% 自动标 ⚠️
- [ ] **judge 校准**：人工评 100 条，与 LLM-judge 算 kappa；< 0.6 则改 rubric 或换 judge
- [ ] 写一份「**评估系统设计文档**」：层级（unit / e2e / online）/ 负责人 / SLO / 更新节奏

### 配套 notebook
⏳ 待建（计划：GitHub Actions eval / judge 校准 / 飞轮自动化）

### 推荐资源
- [Hamel Husain · Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- [Eugene Yan · LLM-Patterns](https://eugeneyan.com/writing/llm-patterns/)
- 论文：[Holistic Evaluation of Language Models (HELM, 2022)](https://arxiv.org/abs/2211.09110)
- 业界：Anthropic / OpenAI 各自公开的 model card

### 能力检验
- [ ] 评估体系能在 1 天告诉团队「上周新版本好坏 + 差在哪类 case」
- [ ] 团队里没有人会跳过 eval 直接上线
- [ ] 给一个完全没评估的项目，1 周内搭好端到端 + 可观测最小可用方案

---

## 与其它章节的关系

- **强耦合 / 贯穿**：本章应**贯穿 01-04 的所有 stage**
  - [01-RAG](../01-RAG/) 每段优化都要先有评估
  - [02-Agent](../02-Agent/) trajectory 评估、tool 成功率
  - [04-微调](../04-模型微调-Finetuning/) 训前后 baseline 必须对比；DPO 偏好数据来源就是评估失败 case
- **下游被引**：[06-部署](../06-工程化与部署-Deployment/) 可观测是部署的一部分，不是事后补的
- **上游依赖**：无

---

## 反模式

- ❌ **「我看了 5 个例子，感觉好了」**：5 个不足以下结论，至少 30
- ❌ **每次手动跑评估**：必然不会持续，必须脚本化 + CI
- ❌ **用模型评估自己**：偏见严重，至少用更强模型或多模型投票
- ❌ **指标只看均值**：分布更重要 —— 10% 的灾难性失败被 90% 的小胜淹没了
- ❌ **没有反例**：所有题都是「应该答 X」，没有「应该拒答」题，模型学得撒谎
- ❌ **eval set 跟代码不在一个仓**：随便丢，过两月找不到

---

## 前沿追踪

- 关注：Anthropic / OpenAI / Google DeepMind 的 model card + evaluation paper
- 关注：[HELM](https://crfm.stanford.edu/helm/)、[BIG-Bench](https://github.com/google/BIG-bench)、[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- 关注：Eugene Yan、Hamel Husain、Jason Liu 的实战文章
- 实践规律：**评估系统投入应该和应用本身投入持平**
