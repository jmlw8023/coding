# 06-工程化与部署-Deployment · Practice Lab

> 配套 [06-工程化与部署-Deployment/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `rag` conda env（同 01-RAG 的 kernel）。

---

## 环境前置

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/06-工程化与部署-Deployment/practice"
jupyter lab
# 启动后选 "Python (rag)" kernel
```

---

## 本机环境（实测）

- rag env + rag_project 可用
- FastAPI + structlog 可用
- Docker + Docker Compose 可用
- k8s（minikube/kind）可选

---

## 学习顺序

每个 notebook 结构：**学习目标 → 阶段拆解 → 深入思考 → 自检 ✅ → 下一步**。

### Stage 1 · 入门（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 47 | [stage1_入门/47_fastapi_rag_service.ipynb](stage1_入门/47_fastapi_rag_service.ipynb) | FastAPI / structlog / pydantic-settings | 90 min |

### Stage 2 · 进阶（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 48 | [stage2_进阶/48_async_sse_docker.ipynb](stage2_进阶/48_async_sse_docker.ipynb) | async 全链路 / SSE 流式 / Docker Compose | 90 min |

### Stage 3 · 高级（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 49 | [stage3_高级/49_vllm_multireplica_caching.ipynb](stage3_高级/49_vllm_multireplica_caching.ipynb) | vLLM / prefix cache / 多副本 / 缓存 + 熔断 | 90 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 50 | [stage4_专家/50_k8s_sre_data_flywheel.ipynb](stage4_专家/50_k8s_sre_data_flywheel.ipynb) | k8s / CI/CD / SLO / 数据飞轮 | 90 min |

---

## 学完之后

回到 [06-工程化与部署-Deployment/README.md](../README.md) 打 checkbox + 做能力检验。Stage 1–4 全部通过 → 可进 [07-Capstone](../07-综合项目-Capstone/)。
