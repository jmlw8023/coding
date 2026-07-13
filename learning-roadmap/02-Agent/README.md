# 02 - Agent · 智能体

> **一句话定位**：让 LLM 自主决定下一步动作（调工具 / 写答案 / 调子 Agent），把一次性问答升级为多步任务执行。
> **时间预算**：4–10 周（与 01 章可并行；建议先完成 01 入门）。
> **适用判断**：能独立设计多 Agent 协作、能在 1 周交付带评估的领域 Agent → 进 [03-Skills](../03-Claude-Skills/) 或 [07-Capstone 项目 2](../07-综合项目-Capstone/)。

---

## 学习目标

完成本章后，你应当：
1. 不查文档 10 分钟白板画出 Agent loop 的 5 步
2. 不依赖框架，200 行内手撸一个带工具调用的最小 Agent
3. 能讲清「裸写 / LangChain / LangGraph / CrewAI / Claude Agent SDK」各自适合什么场景
4. 能讲清 MCP 是什么、解决什么问题、怎么写最小 MCP server
5. 给一个 Agent 任务，能在 8 小时内做出带 trace + 错误恢复的工程化版本

---

## 实战状态

✅ **已落地 8 个 notebook**（4 stage 全覆盖，全部 smoke 通过）→ [practice/](practice/)

所有 notebook **offline-first**（规则 stub LLM），`MODE='ONLINE'` 时切到本机 Ollama OpenAI 兼容接口。

---

## 全章地图

```
       ┌────────────────────────────────────────┐
       │              用户目标                    │
       └───────────────────┬────────────────────┘
                           ▼
       ┌────────────────────────────────────────┐
       │   Agent Loop                            │
       │   ┌────────────────────────────────┐    │
       │   │ 1. 观察当前状态/上下文           │    │
       │   │ 2. 思考下一步（LLM 推理）         │    │
       │   │ 3. 调用工具 / 输出答案           │    │
       │   │ 4. 把结果加回上下文              │    │
       │   │ 5. 判断是否完成，未完成则回到 1   │    │
       │   └────────────────────────────────┘    │
       └───────────────────┬────────────────────┘
                           ▼
       ┌────────────────────────────────────────┐
       │              最终结果                    │
       └────────────────────────────────────────┘
```

记住这张图。所有 ReAct / Tool Use / MCP / LangGraph / AutoGen，**本质上都是在解决这五步里某一步的工程问题**。

---

## 阶段 1 · 入门：理解 Agent 的最小骨架（1–2 周）

### 核心知识点
- **什么是 Agent**：不是「会聊天的 LLM」，是「能自主决定下一步动作」的 LLM 系统
- **ReAct 范式**：Reason + Act 交替（论文级原点）
- **Function Calling / Tool Use**：让 LLM 输出**结构化工具调用请求**而不是自然语言
- **System Prompt 的作用**：约束 Agent 的人格、能力边界与工作流
- **最小三件套**：System Prompt + 工具列表 + 循环

### 动手任务
- [ ] **不用任何框架**，写 200 行内 Agent，含 2 个 tool（`get_current_time()` / `calculate(expr)`）
- [ ] 用 Anthropic SDK 的 Tool Use 重写一遍，对比代码量与稳定性
- [ ] 读完 [ReAct 论文 (2022)](https://arxiv.org/abs/2210.03629)，能口头复述「为什么 Reason+Act 比纯 Reason 更好」
- [ ] **观察 Claude Code 自身**：跑 `/help`，思考每个工具在 Agent 框架里扮演什么角色

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 25 | [stage1_入门/25_minimal_agent.ipynb](practice/stage1_入门/25_minimal_agent.ipynb) | 200 行手撸 ReAct Agent + 2 工具 |
| 26 | [stage1_入门/26_tool_use_openai_style.ipynb](practice/stage1_入门/26_tool_use_openai_style.ipynb) | OpenAI/Anthropic 风格 Tool Use 规范 + parallel |
| 27 | [stage1_入门/27_observe_claude_code.ipynb](practice/stage1_入门/27_observe_claude_code.ipynb) | 拆 Claude Code 的 Agent loop + trace 甘特图 |

### 推荐资源
- [Anthropic · Building effective agents](https://www.anthropic.com/research/building-effective-agents) —— **本章核心读物，必读多遍**
- [Anthropic · Tool Use 文档](https://docs.anthropic.com/claude/docs/tool-use)
- [Lilian Weng · LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- 论文：[ReAct (2022)](https://arxiv.org/abs/2210.03629)、[Toolformer (2023)](https://arxiv.org/abs/2302.04761)

### 能力检验
- [ ] 不查文档 10 分钟内白板画出 Agent loop 的 5 步
- [ ] 现场写一个最小 Agent，调用 1 个虚构工具
- [ ] 解释「为什么 Function Calling 比让 LLM 输出 JSON 更可靠」

---

## 阶段 2 · 进阶：工具集、记忆、错误处理、MCP（2–3 周）

### 核心知识点
- **工具设计 4 原则**：小而专 / 可观测 / 可幂等 / 错误明确
- **记忆系统**：短期（上下文）/ 长期（向量库 / KV）/ 工作记忆（scratchpad）/ 总结型
- **控制策略**：最大步数 / 重试与超时 / token 预算 / 人类介入点 (HITL)
- **MCP（Model Context Protocol）**：把工具/资源/prompt 标准化暴露给任何 LLM 客户端 —— 「Agent 的 USB」
- **主流框架对比表**：

| 框架 | 优点 | 缺点 | 适合 |
|------|------|------|------|
| 裸写 + SDK | 完全可控、易 debug | 样板代码多 | 学习 / 一次性项目 |
| LangChain Agents | 工具生态丰富 | 抽象多、API 跟不上版本 | 快速原型 |
| LangGraph | 显式状态机、可观察 | 学习曲线陡 | 复杂工作流 |
| CrewAI | 多 Agent 协作开箱即用 | 自由度低 | 角色分工型任务 |
| AutoGen | 微软出品、对话式 | 文档难读 | 研究实验 |
| **Claude Agent SDK** | 与 Claude Code 同源 | 仅 Anthropic | Anthropic 生态 |

### 动手任务
- [ ] 改造入门 Agent：加 `read_file` / `write_file` / `search_web`（或 mock），设最大 10 步、token 预算监控
- [ ] **接入 MCP**：本机为 Claude Code 配置一个 MCP server（哪怕是别人写的 filesystem server），观察工具被发现 + 调用的过程
- [ ] 写 Agent 评估题集：10 个不同复杂度任务，分别用「裸写 / LangChain」跑，记录成功率/步数/token
- [ ] 读 [ragflow/agent/](../../ragflow/agent) 源码，看企业 Agent 如何与 RAG 协作
- [ ] 加长期记忆：跑完的对话存入向量库，下次开新会话基于 query 召回最近相关历史

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 28 | [stage2_进阶/28_multi_tool_memory_budget.ipynb](practice/stage2_进阶/28_multi_tool_memory_budget.ipynb) | 4 工具 + 长期记忆（向量库召回）+ token/步数预算 |
| 29 | [stage2_进阶/29_minimal_mcp_server.ipynb](practice/stage2_进阶/29_minimal_mcp_server.ipynb) | 手撸 MCP-like 协议（initialize / tools/list / tools/call） |

### 推荐资源
- [Anthropic · MCP 介绍](https://www.anthropic.com/news/model-context-protocol)
- [MCP 官方文档](https://modelcontextprotocol.io/)
- [LangChain Agents 文档](https://python.langchain.com/docs/concepts/agents/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— 重点看 state graph
- [CrewAI 文档](https://docs.crewai.com/)

### 能力检验
- [ ] 给一个 Agent 任务（如「帮我整理桌面 PDF 按主题分类」），2 小时做出能跑版本含错误处理
- [ ] 讲清「为什么需要 MCP」，10 分钟写最小 MCP server
- [ ] 看一段 LangGraph 代码，立刻指出哪些是 state / node / edge

---

## 阶段 3 · 高级：多 Agent、规划、工作流编排（2–3 周）

### 核心知识点
- **多 Agent 协作模式**：
  - Manager-Worker（Claude Code sub-agent 这套）
  - Pipeline（A 输出 → B 输入）
  - Debate / Critique（两个 Agent 互相 review）
  - Mixture-of-Agents（多投票 / 加权聚合）
- **任务规划**：Plan-then-Execute / Dynamic Planning / HTN；**何时该 plan / 何时不该**
- **状态与控制流**：显式 state machine（LangGraph）/ 检查点 / 回滚 / 分支与合并
- **工具生态**：浏览器自动化（Playwright / Browser Use）/ 代码沙箱（Modal / E2B）/ 桌面控制（Anthropic Computer Use）
- **评估 Agent 的难点**：不只看最终对不对，还要看 trajectory；用 LLM-as-judge 做轨迹评

### 动手任务
- [ ] **Manager + 3 Worker** 系统：给定长论文，3 Worker 分别写「摘要 / 评价 / 类比解释」，Manager 汇总成博客
- [ ] **复刻 Claude Code 的 sub-agent 思想**：能 spawn 子 Agent 处理独立任务，理解上下文隔离的价值
- [ ] **接入浏览器**：用 [Browser Use](https://github.com/browser-use/browser-use) 或 Playwright 完成「访问网站 → 填表 → 截图」
- [ ] 用 [ragflow/agent](../../ragflow/agent) 参考重新设计 [rag_project/](../../rag_project/)，让它能「自主决定要不要检索 / 是否再问一次」
- [ ] 做 Agent 测试集：20 任务，每个有成功标准 + 禁止行为，用 LLM-judge 自动打分

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 30 | [stage3_高级/30_manager_worker_multiagent.ipynb](practice/stage3_高级/30_manager_worker_multiagent.ipynb) | Manager + 3 Workers 协作 + 失败率叠加演示 |
| 31 | [stage3_高级/31_state_machine_langgraph.ipynb](practice/stage3_高级/31_state_machine_langgraph.ipynb) | LangGraph 状态机 + checkpoint + interrupt(HITL) |

### 推荐资源
- 论文：[AutoGen (2023)](https://arxiv.org/abs/2308.08155)、[Reflexion (2023)](https://arxiv.org/abs/2303.11366)、[Voyager (2023)](https://arxiv.org/abs/2305.16291)、[Mixture-of-Agents (2024)](https://arxiv.org/abs/2406.04692)
- [Anthropic · Computer Use](https://docs.anthropic.com/claude/docs/computer-use)
- [Browser Use](https://github.com/browser-use/browser-use)
- [LangGraph 多 Agent 教程](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Andrew Ng · Agentic Workflows 系列](https://www.deeplearning.ai/the-batch/issue-241/)

### 能力检验
- [ ] 白板设计「中等复杂度任务」的多 Agent 架构，列出每 Agent 的输入/输出/失败模式
- [ ] 区分「单 Agent + 工具」vs「多 Agent 协作」何时该用
- [ ] 写过的 Agent 能从失败步骤恢复继续，不重启

---

## 阶段 4 · 专家：自研 Agent 框架、可控性、安全、生产化（持续）

### 核心知识点
- **自研框架抽象**：Tool / Memory / Planner / Executor / Observer；可插拔 backend；完整 trace / replay；配置即代码
- **可控性与安全**：沙箱化、权限分级、审计日志、prompt injection 防护、资源限制
- **生产化**：异步队列（Celery / Arq / Temporal）/ 长任务进度 / 中断 / 多租户与计费 / 与 [06-部署](../06-工程化与部署-Deployment/) 深度交叉
- **持续优化**：失败 trajectory 收集 → 训 reward model → DPO/RLAIF 优化 Agent 行为（与 [04-微调](../04-模型微调-Finetuning/) 交叉）

### 动手任务
- [ ] 从零写 mini Agent 框架（500 行内），含 tool 注册 / memory 接口 / 可插拔 LLM / trace。同一份代码跑通至少 3 种 backend
- [ ] 设计生产 Agent 服务：FastAPI + Celery + Redis，并发 10 长任务，支持中断/进度查询/结果回查
- [ ] Agent 安全审计：构造 5 种 prompt injection 攻击，看 Agent 会不会上当；针对每种设计防御
- [ ] 训一个 Agent 行为优化器（与 [04 章](../04-模型微调-Finetuning/) 联动）：用失败/成功 trajectory 做 SFT 或 DPO
- [ ] 写一份「Agent 系统设计文档」

### 配套 notebook
| # | 文件 | 主题 |
|---|------|------|
| 32 | [stage4_专家/32_mini_agent_framework.ipynb](practice/stage4_专家/32_mini_agent_framework.ipynb) | ~500 行自研 mini framework + 5 种 prompt injection 与防御 |

### 推荐资源
- [Anthropic · Claude Agent SDK 文档](https://docs.anthropic.com/)
- 论文：[Agent Workflow Memory (2024)](https://arxiv.org/abs/2409.07429)、[OpenDevin (2024)](https://arxiv.org/abs/2407.16741)
- 开源参考：[microsoft/autogen](https://github.com/microsoft/autogen)、[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)、[All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)
- 安全：[OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

### 能力检验
- [ ] 8 小时内从零做出能用的领域 Agent（如「整理 Slack 收件箱」），含部署/监控/人工介入
- [ ] 能审一个别人 Agent 系统，给出 5 个安全/可靠性问题
- [ ] 能讲 1 小时「为什么不要追新 Agent 框架」分享课

---

## 与其它章节的关系

- **上游依赖**：[00-基础 stage 1](../00-基础-Foundations/) 的 asyncio / 装饰器
- **强耦合**：[03-Claude Skills](../03-Claude-Skills/) —— Skills 是「Agent 工作流的封装与复用」
- **下游被引**：
  - [01-RAG stage 3](../01-RAG/) 的 Agentic RAG（Self-RAG / CRAG）就是 Agent 思想在 RAG 里的渗透
  - [04-微调](../04-模型微调-Finetuning/) 可针对 Agent function-calling 行为做 SFT
  - [07-Capstone 项目 2](../07-综合项目-Capstone/) 是本章的综合演练
- **并行可学**：[05-评估](../05-评估与可观测性-Evaluation/) 的 trajectory eval

---

## 反模式

- ❌ **直接上多 Agent**：单 Agent 都没调好不要碰多 Agent。多 Agent 失败率是单 N 倍
- ❌ **工具描述偷懒**：`tool_a` / `do_thing` 这种命名，再贵的 LLM 也救不了
- ❌ **没有最大步数**：Agent 会陷入死循环，烧光 token 配额
- ❌ **把所有工具塞进 system prompt**：50 个工具的 prompt 召回准确率会崩，考虑工具检索
- ❌ **不打 trace 就发布**：生产出问题时没 trace 等于盲飞
- ❌ **追热点框架**：每周换一个，从没把任何一个推到生产

---

## 前沿追踪

- 论文 feed：[arXiv cs.AI](https://arxiv.org/list/cs.AI/recent) 「Agent」关键词
- 必看团队：Anthropic、Adept、Cognition (Devin)、Voyager 团队、Princeton SHIM lab
- 必看博客：Anthropic Engineering、LangChain Blog、Sierra
- 关键基准：SWE-bench、WebArena、GAIA
