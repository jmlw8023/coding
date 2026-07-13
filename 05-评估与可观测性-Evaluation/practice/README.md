# 05-评估与可观测性-Evaluation · Practice Lab

> 配套 [05-评估与可观测性-Evaluation/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `rag` conda env（同 01-RAG 的 kernel）。

---

## 环境前置

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/05-评估与可观测性-Evaluation/practice"
jupyter lab
# 启动后选 "Python (rag)" kernel
```

---

## 本机环境（实测）

- rag env + rag_project 可用
- Langfuse 可用 Docker compose 一键启动（`docker compose up -d langfuse`）
- 真实 LLM-as-jude 需要调用 Claude/GPT-4 API
- A/B 测试需要模拟用户流量

---

## 学习顺序

每个 notebook 结构：**学习目标 → 阶段拆解 → 深入思考 → 自检 ✅ → 下一步**。

### Stage 1 · 入门（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 43 | [stage1_入门/43_basic_metrics_and_eval_set.ipynb](stage1_入门/43_basic_metrics_and_eval_set.ipynb) | 基础指标 / eval set 设计 / baseline 表 | 75 min |

### Stage 2 · 进阶（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 44 | [stage2_进阶/44_llm_judge_and_ragas.ipynb](stage2_进阶/44_llm_judge_and_ragas.ipynb) | LLM-as-judge / RAGAS 5 指标 / 分层评估 | 90 min |

### Stage 3 · 高级（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 45 | [stage3_高级/45_ab_testing_and_observability.ipynb](stage3_高级/45_ab_testing_and_observability.ipynb) | A/B 测试 / Langfuse 接入 / 人工评测 | 90 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 46 | [stage4_专家/46_data_flywheel_and_ci_eval.ipynb](stage4_专家/46_data_flywheel_and_ci_eval.ipynb) | 数据飞轮 / CI 集成 / judge 校准 | 90 min |

---

## 学完之后

回到 [05-评估与可观测性-Evaluation/README.md](../README.md) 打 checkbox + 做能力检验。Stage 1–4 全部通过 → 可进 [06-部署](../06-工程化与部署-Deployment/) 或 [07-Capstone 项目 3](../07-综合项目-Capstone/)。
