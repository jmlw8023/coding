# 02-Agent · Practice Lab

> 配套 [02-Agent/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `rag` conda env（同 [00-基础/practice](../../00-基础-Foundations/practice/README.md) 的 kernel）。

---

## 环境前置

如果 Foundations 章节走完，jupyter + `rag` kernel 已就绪。否则回 [00-基础/practice/README.md](../../00-基础-Foundations/practice/README.md) 装。

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/02-Agent/practice"
jupyter lab
# 启动后选 "Python (rag)" kernel
```

---

## 双模式约定（沿用 01-RAG）

每个 notebook 顶部 `MODE` 开关：

| MODE | 行为 | 何时用 |
|------|------|--------|
| `OFFLINE` | LLM 用规则模拟（hash 路由 / 关键词匹配） | **默认**，永远跑得通 |
| `ONLINE` | 调本机 Ollama via OpenAI 兼容接口 `/v1/chat/completions` | 看真效果；**先 `ollama serve`** |

---

## 已知环境坑（已规避）

- ❌ `anthropic` SDK 没装 → 用 `openai` SDK 走 Ollama 的 OpenAI 兼容接口（API 形状一致，迁移到真 Anthropic 只换 client 一行）
- ❌ `mcp` Python SDK 没装 → 29 号 notebook **手撸**最小 MCP-like JSON-RPC server，教协议本质
- ❌ 浏览器自动化 `playwright` / `browser-use` 没装 → 高级阶段用 stub mock 演示
- ⚠ Ollama 默认不在跑 → `MODE='ONLINE'` 前另起终端 `ollama serve`

---

## 学习顺序

每个 notebook 结构：**学习目标 → 阶段拆解 → 深入思考 → 自检 ✅ → 下一步**。

### Stage 1 · 入门（3 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 25 | [stage1_入门/25_minimal_agent.ipynb](stage1_入门/25_minimal_agent.ipynb) | 200 行手撸 ReAct Agent + 2 工具 | 60 min |
| 26 | [stage1_入门/26_tool_use_openai_style.ipynb](stage1_入门/26_tool_use_openai_style.ipynb) | OpenAI/Anthropic 风格 Tool Use 规范 | 60 min |
| 27 | [stage1_入门/27_observe_claude_code.ipynb](stage1_入门/27_observe_claude_code.ipynb) | 拆 Claude Code 的 Agent loop + trace 可视化 | 45 min |

### Stage 2 · 进阶（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 28 | [stage2_进阶/28_multi_tool_memory_budget.ipynb](stage2_进阶/28_multi_tool_memory_budget.ipynb) | 4 工具 + 长期记忆（向量库召回）+ token/步数预算 | 90 min |
| 29 | [stage2_进阶/29_minimal_mcp_server.ipynb](stage2_进阶/29_minimal_mcp_server.ipynb) | 手撸 MCP-like 协议（initialize / tools/list / tools/call） | 75 min |

### Stage 3 · 高级（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 30 | [stage3_高级/30_manager_worker_multiagent.ipynb](stage3_高级/30_manager_worker_multiagent.ipynb) | Manager + 3 Workers 协作写综述 | 90 min |
| 31 | [stage3_高级/31_state_machine_langgraph.ipynb](stage3_高级/31_state_machine_langgraph.ipynb) | LangGraph 状态机 + 检查点 + 中断恢复 | 90 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 32 | [stage4_专家/32_mini_agent_framework.ipynb](stage4_专家/32_mini_agent_framework.ipynb) | ~500 行自研 mini framework + 5 种 prompt injection 与防御 | 120 min |

---

## 学完之后

回到 [02-Agent/README.md](../README.md) 打 checkbox + 做能力检验。Stage 1–4 全部通过 → 可进 [03-Skills](../../03-Claude-Skills/) 或 [07-Capstone 项目 2](../../07-综合项目-Capstone/)。
