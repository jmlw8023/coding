# 06 - 工程化与部署 Deployment

> **一句话定位**：把 RAG / Agent / 微调模型从「本地能跑」推进到「线上多副本 + 监控 + 能回滚」的生产服务。
> **时间预算**：2–8 周。
> **适用判断**：能为 LLM 服务从零到上线交付完整方案（架构 + 编排 + 监控 + 灰度 + SLO + 回滚演练）→ 进 [07-Capstone](../07-综合项目-Capstone/)。

---

## 学习目标

完成本章后，你应当：
1. 半小时内把任何 Python 函数包成带 schema 的 FastAPI endpoint
2. 一行命令把整个 RAG 系统在新机器上跑起来（Docker compose）
3. 单机 GPU 下 8 小时内把 LLM 服务吞吐做到接近硬件理论上限
4. 在压测下能定位瓶颈是「网络 / CPU / GPU / 显存 / 锁竞争」中的哪一类
5. 一次真实故障里 30 分钟内定位 + 缓解 + 写出 postmortem

---

## 实战状态

⏳ **notebook 待建**（路线已写）。[01-RAG 24 号 notebook](../01-RAG/practice/stage4_专家/24_eval_and_serving.ipynb) 已演示「RAG 类封装 + stdlib HTTP server」最小骨架，本章在此之上展开 FastAPI / Docker / vLLM / k8s。

---

## 全章地图

```
┌────────────────────────────────────────────────────┐
│   单机原型 → API 服务 → 容器化 → 多副本 → 弹性 → 飞轮  │
└────────────────────────────────────────────────────┘
       │           │         │        │       │       │
       ▼           ▼         ▼        ▼       ▼       ▼
    Python      FastAPI   Docker   k8s/   GPU autoscale + cost
    脚本         + uv      compose   nomad   continuous batching
                 异步                 vLLM    数据回流到训练
                 流式                 TGI
```

---

## 阶段 1 · 入门：把脚本包成 HTTP 服务（半–1 周）

### 核心知识点
- **API 框架选型**：FastAPI（首选）/ Flask（旧）/ Starlette（底层）
- **同步 vs 异步**：LLM 调用必须异步，否则一个请求阻死整个进程
- **请求生命周期**：parse → validate (pydantic) → handler → return
- **环境管理**：`.env` + `pydantic-settings`，秘密绝不入 git
- **结构化日志**：JSON 格式 + 关键字段（request_id / user_id / latency）

### 动手任务
- [ ] **把 [rag_project/](../../rag_project/) 包成 FastAPI**：
  - 新增 `src/rag/api.py`
  - 暴露 `POST /query` / `POST /docs` / `DELETE /docs/{id}` / `GET /healthz`
  - 用 pydantic 定义请求/响应 schema
  - `uvicorn rag.api:app --reload` 起服务
- [ ] 用 `httpx` 或 curl 调通所有 endpoint
- [ ] 接结构化日志（`structlog` 或 `loguru`），每条 log 含 `request_id` / `latency_ms` / `tokens`
- [ ] 加 `.env.example` + `pydantic-settings` 配置加载

### 配套 notebook
⏳ 待建（计划：FastAPI 包 rag_project / structlog / pydantic-settings）

### 推荐资源
- [FastAPI 官方教程](https://fastapi.tiangolo.com/tutorial/) —— 前 5 章足够
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [structlog](https://www.structlog.org/)
- 本机：[rag_project/src/rag/](../../rag_project/src/) 已有 CLI，借鉴改造

### 能力检验
- [ ] 半小时内把任何 Python 函数包成带 schema 的 FastAPI endpoint
- [ ] 解释「为什么 LLM 服务一定要用 async」
- [ ] 现场演示：日志能根据 request_id 串起一个请求的完整生命周期

---

## 阶段 2 · 进阶：异步 / 流式 / 容器化（1–2 周）

### 核心知识点
- **真正的异步**：`async def` + `await` 全链路；HTTP `httpx.AsyncClient`（`requests` 是 trap）；LLM async；DB / 向量库选 async；`asyncio.Semaphore` 限并发
- **流式响应**：`StreamingResponse` + async generator；SSE / WebSocket / chunked HTTP；LLM `stream=True` 转发；流到一半挂了怎么办
- **任务编排**：短（< 30s）直接同步；中（30s–几分钟）返回 task_id + 轮询/SSE；长（小时级）Celery / Arq / Temporal
- **容器化**：Dockerfile 多阶段构建；`.dockerignore` + 瘦身；`docker-compose.yml` 编 app + vector_db + redis；GPU 容器 `--gpus all`
- **配置 / 秘密**：12-factor；`.env` for dev、k8s Secret / Vault for prod；模型权重的挂载策略（不要塞镜像）

### 动手任务
- [ ] **rag_project 改全 async**：sync → async，benchmark QPS 对比
- [ ] **加流式响应**：`/query/stream` 用 SSE 把 LLM token 实时推前端，写最简 HTML 客户端
- [ ] **写 Dockerfile + docker-compose**：app（FastAPI）+ chromadb（持久卷）+ ollama（可选 GPU）+ redis；`docker compose up` 一条命令
- [ ] **加 embedding 缓存**：相同文本不重算 embedding，Redis `SHA1(text) → vector`，看命中率
- [ ] **加任务队列**：用 `arq`（轻量 async-friendly Celery）把「增量入库」改异步，返回 job_id

### 配套 notebook
⏳ 待建（计划：async 改造 / SSE / Docker compose / 缓存 + arq）

### 推荐资源
- [Asyncio Cheat Sheet](https://docs.python.org/3/library/asyncio-task.html)
- [FastAPI Async 文档](https://fastapi.tiangolo.com/async/)
- [Docker 官方教程](https://docs.docker.com/get-started/)
- [arq](https://arq-docs.helpmanual.io/) / [Celery](https://docs.celeryq.dev/)
- [Real Python · Async IO](https://realpython.com/async-io-python/)

### 能力检验
- [ ] 一行命令把整个 RAG 系统在新机器上跑起来
- [ ] QPS（同硬件下）比 sync 版本至少高 5×
- [ ] 讲清「为什么 LLM 服务的流式不只是用户体验，还能省 token / 早失败」

---

## 阶段 3 · 高级：推理优化、多副本、负载与缓存（1–2 周）

### 核心知识点
- **推理引擎选型**：
  | 引擎 | 适合 | 不适合 |
  |------|------|--------|
  | [vLLM](https://github.com/vllm-project/vllm) | GPU、高吞吐 | CPU-only |
  | [TGI](https://github.com/huggingface/text-generation-inference) | HF 生态、易用 | 极致性能 |
  | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 极致 | 跨厂商 |
  | [llama.cpp / Ollama](https://github.com/ggerganov/llama.cpp) | CPU、GGUF、本地 | 大规模 GPU |
  | [SGLang](https://github.com/sgl-project/sglang) | 结构化输出 / agent | 早期生态 |
- **vLLM 关键概念**：PagedAttention（KV 分页省显存）/ Continuous batching（吞吐 +10×）/ Speculative decoding / **Prefix caching**（对 RAG 巨大收益）
- **多副本与负载均衡**：单机多卡 tensor parallel / 多机多副本 nginx / traefik / k8s service；健康检查 `/healthz` vs `/readyz`
- **缓存层级**：答案 / embedding / prefix / rerank 四层
- **限流与降级**：每用户 QPS / token / 全局并发；高负载跳过 rerank / 用更小模型 / 返回缓存；熔断快速失败

### 动手任务
- [ ] **部署 vLLM 服务**（如有 GPU）：serve 一个 Qwen-7B，对比 Ollama 同硬件吞吐
- [ ] **接 prefix cache**：RAG system prompt 设稳定前缀，看 prefix cache 命中率
- [ ] **多副本演练**：docker-compose 起 2 个 app + nginx，模拟 100 并发，看延迟分布
- [ ] **加全套缓存**：答案 / embedding / rerank 三层接入 Redis，benchmark 命中率与响应时间
- [ ] **限流 + 熔断**：`slowapi` 或 nginx 加每 IP 限流；`tenacity` 给 LLM 调用加重试 + 指数退避

### 配套 notebook
⏳ 待建（计划：vLLM serve / prefix cache 实验 / nginx 多副本 / Redis 缓存）

### 推荐资源
- [vLLM 文档](https://docs.vllm.ai/) / [TGI 文档](https://huggingface.co/docs/text-generation-inference/)
- [Anthropic · Prompt Caching](https://docs.anthropic.com/claude/docs/prompt-caching)
- [Cloudflare · Rate Limiting Patterns](https://blog.cloudflare.com/counting-things-a-lot-of-different-things/)
- 论文：[vLLM/PagedAttention (2023)](https://arxiv.org/abs/2309.06180)

### 能力检验
- [ ] 单机 GPU 下 8 小时把 LLM 服务吞吐做到接近硬件理论上限
- [ ] 讲清「答案缓存的失效策略」（context 哈希、TTL、用户隔离）
- [ ] 在压测下定位瓶颈（网络 / CPU / GPU / 显存 / 锁竞争）

---

## 阶段 4 · 专家：k8s / 弹性 / SRE / 数据飞轮（持续）

### 核心知识点
- **编排平台**：k8s（事实标准）/ Nomad / ECS/GKE/AKS（托管）；LLM 专属 [KServe](https://kserve.github.io/website/) / [BentoML](https://www.bentoml.com/)
- **GPU 调度**：nvidia device plugin / MIG / MPS；调度策略：bin-packing / 抢占；spot 实例 try/retry
- **弹性**：HPA（基于 CPU / 自定义指标 QPS / 队列长度）/ VPA / 冷启动 + pre-warm 池 / 金丝雀 + 蓝绿
- **SRE**：SLO / SLI 定义（「p95 < 3s、可用性 > 99.5%」）/ 错误预算 / on-call / **blameless postmortem**
- **数据飞轮**（与 [05](../05-评估与可观测性-Evaluation/) 交叉）：生产 trace → 失败 → eval / 训练；用户反馈 → DPO；A/B 自动化
- **安全**：API 鉴权（key / OAuth / mTLS）/ prompt injection 防御 / 多租户隔离 / PII 脱敏 + 合规 / 审计日志

### 动手任务
- [ ] **k8s 部署**（minikube / kind 本地）：deployment + service + ingress；HPA 基于 QPS 自动扩缩；GPU resource request
- [ ] **完整 CI/CD**：GitHub Actions lint → test → eval（05 章）→ build image → push → deploy → smoke test；失败自动回滚
- [ ] **SLO 实战**：为 rag_project 定义 3 个 SLO；接 Prometheus + Grafana；超阈值告警到 Slack / 邮件
- [ ] **完整数据飞轮**：Langfuse 失败 trace 自动导出 → 加 eval → CI 验证 → Argilla 标注 → 周期 fine-tune embedding
- [ ] **安全 review**：用 `/security-review` skill 审 API，修掉至少 3 个发现

### 配套 notebook
⏳ 待建（计划：minikube 部署 / GitHub Actions / Prometheus 接入）

### 推荐资源
- [Kubernetes 官方教程](https://kubernetes.io/docs/tutorials/)
- [KServe / BentoML](https://www.bentoml.com/)
- 《Site Reliability Engineering》(Google) —— 免费 PDF
- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Chip Huyen · ML in Production 系列](https://huyenchip.com/blog/)

### 能力检验
- [ ] 为 LLM 服务从零到上线交付完整方案（架构 + 编排 + 监控 + 灰度 + SLO + 回滚演练）
- [ ] 一次真实故障里 30 分钟内定位 + 缓解 + 写出 postmortem
- [ ] 能向团队解释「为什么 LLM 不能像普通微服务那样部署」

---

## 与其它章节的关系

- **上游依赖**：
  - [01-RAG](../01-RAG/) / [02-Agent](../02-Agent/) / [04-微调](../04-模型微调-Finetuning/) 都是本章的部署对象
  - 本章里 vLLM / 量化部分依赖 [04 stage 4](../04-模型微调-Finetuning/) 的推理优化知识
- **强耦合**：[05-评估](../05-评估与可观测性-Evaluation/) —— 本章可观测部分就是评估的在线段
- **下游被引**：[07-Capstone](../07-综合项目-Capstone/) 所有项目最后都要上本章流程

---

## 反模式

- ❌ **直接把 jupyter notebook 包成服务**：不是产品
- ❌ **同步阻塞 LLM 调用**：第一个并发请求就完了
- ❌ **模型权重塞 Docker 镜像**：镜像 20GB，部署慢到崩，挂卷
- ❌ **没限流**：被恶意/善意刷一波直接被薅光配额
- ❌ **没 healthcheck**：k8s 不知道你挂了，流量继续打
- ❌ **日志没 request_id**：故障调查像大海捞针
- ❌ **直连数据库做 vector search 不加 timeout**：一次慢查询拖死所有 worker

---

## 前沿追踪

- 推理优化论文：vLLM、SGLang、TensorRT-LLM 团队
- 开源：vllm-project、infiniflow、bentoml、langfuse
- 业界博客：Anyscale、Databricks、Modal、Replicate、Together AI
- 大会：Ray Summit、KubeCon、MLOps Community
