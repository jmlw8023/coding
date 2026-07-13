# 03 - Claude Skills · 把工作流封装成可复用能力

> **一句话定位**：把重复执行的「分析 / 重构 / 验证 / 部署」流程，封装成 Claude Code 可以自动调用或 `/skill` 触发的 Skill。
> **时间预算**：1–3 周。
> **适用判断**：本机 `~/.claude/skills/` 下已有 5+ 个自写 Skill 在长期工作里用 → 进 [07-Capstone 项目 2](../07-综合项目-Capstone/)。

---

## 学习目标

完成本章后，你应当：
1. 一句话解释 Skill / sub-agent / MCP / slash command / settings hooks 各自管什么
2. 不查文档写出一个 Skill（含 frontmatter + 步骤 + 触发示例 + 反例），description 召回准确
3. 设计 Skill 时知道「单一职责」「显式触发条件」「显式不触发条件」三铁律
4. 能为新项目在 1 小时内写 3 个有用的 Skill
5. 能向不熟悉 Claude Code 的同事讲清「Skill 与 CLAUDE.md 各自适合放什么」

---

## 实战状态

✅ **notebook 已就位**（路线已写）。本章动手任务以「在 `~/.claude/skills/` 下真造 Skill 并跑通」为主，notebook 用来教学 + 测 description 召回率。

### Notebook 列表（6 个，4 个 stage）
| Stage | Notebook | 状态 |
|-------|----------|------|
| 1 入门 | [33_skill_browser.ipynb](practice/stage1_入门/33_skill_browser.ipynb) | ✅ 完成（沙箱建模） |
| 2 进阶 | [34_first_skill.ipynb](practice/stage2_进阶/34_first_skill.ipynb) | ✅ 完成（写 Skill + 反例） |
| 2 进阶 | [35_description_recall_test.ipynb](practice/stage2_进阶/35_description_recall_test.ipynb) | ✅ 完成（10 触发 / 10 不触发） |
| 3 高级 | [36_skill_subagent_compose.ipynb](practice/stage3_高级/36_skill_subagent_compose.ipynb) | ✅ 完成 |
| 3 高级 | [37_skill_hooks_permissions.ipynb](practice/stage3_高级/37_skill_hooks_permissions.ipynb) | ✅ 完成 |
| 4 专家 | [38_skill_repo_skeleton.ipynb](practice/stage4_专家/38_skill_repo_skeleton.ipynb) | ✅ 完成 |

所有 notebook 均遵循 **intro → setup → sN-md/cN → deep → checklist** 模式。

---

## Skill 是什么 · 一句话解释

**Skill = 一份带 YAML frontmatter 的 markdown 指南 + 可选脚本/资源**，存放在 `~/.claude/skills/<name>/SKILL.md`。

Claude Code 会按 description 自动召回；你也可以 `/skill-name` 显式触发。

```
┌────────────────────────────────────────────────┐
│   你日常重复做的工作                            │
│   "review 这个 PR" / "在项目里加测试" / ...      │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│   写成 SKILL.md：                                │
│   - 名字 / 描述 / 触发条件（YAML frontmatter）    │
│   - 何时使用 / 何时不使用                         │
│   - 步骤 / 工具 / 示例                            │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│   下次再遇到这类任务：                            │
│   - Claude 自动建议 / 自动调用                    │
│   - 或你打 /skill-name 触发                       │
└────────────────────────────────────────────────┘
```

---

## 全章地图

```
   Skill = workflow 的固化封装
        │
        ├── description = 召回钩子（写不好 = 永远召不回）
        ├── 步骤   = 流程
        ├── 资源   = 模板 / 脚本
        └── 反例   = 防误触发
```

---

## 阶段 1 · 入门：用别人写好的 Skill（半周）

### 核心知识点
- Claude Code 默认带的 Skill 列表（开 Claude Code，看 system reminder 区域）
- 触发方式：自动（description 匹配）vs 显式（`/skill-name`）
- Skill 与其它机制的关系：
  - **Skill** = 你定义的「做某类事的标准流程」
  - **sub-agent** = 隔离的执行环境（可被 Skill 调用）
  - **slash command** = 在 CLI 里手动触发某能力
  - **MCP** = 给 Claude 暴露外部工具/资源的协议
  - **settings hooks** = 在 tool 调用前/后跑的脚本

### 动手任务
- [ ] 列出本机已有的所有 Skill（看 system reminder），写一份个人速查表
- [ ] 选 5 个 Skill 各跑一次：`/init` / `/code-review` / `/security-review` / `/verify` / `/fewer-permission-prompts`
- [ ] 浏览本机 `C:\Users\ril\.claude\` 目录，搞清楚 Skill 文件放在哪、长什么样、与 `settings.json` 的关系

### 配套 notebook
✅ **[33_skill_browser.ipynb](practice/stage1_入门/33_skill_browser.ipynb)** — Skill 机制 / 沙箱建模 / SkillLoader 实现

### 推荐资源
- 本机：`~/.claude/` 目录（`skills/`、`settings.json`、`commands/`）
- [Anthropic · Claude Code 文档](https://docs.anthropic.com/claude/docs/claude-code)

### 能力检验
- [ ] 给同事讲清「什么时候 Claude 自动建议 skill / 什么时候你要手动 `/skill-name`」
- [ ] 列出 3 个你想自动化的日常任务（候选 Skill 主题）

---

## 阶段 2 · 进阶：写第一个 Skill（半–1 周）

### 核心知识点
- **SKILL.md 结构**：
  ```markdown
  ---
  name: my-skill                       # 唯一短名
  description: 一句话说清「何时调用」    # 决定自动召回准确率
  ---

  # 何时使用
  （具体场景，举例越多越好）

  # 不要在这些时候使用
  （避免误触发）

  # 步骤
  1. ...

  # 示例
  （输入 → 输出对）
  ```
- **description 的关键作用**：这是 Claude 决定「要不要触发你 Skill」的唯一依据
  - 反例：`description: "处理代码相关的事情"` —— 太泛
  - 正例：`description: "When the user asks to run / verify / smoke-test the local app to confirm a change works"`
- **资源文件**：Skill 目录下可放 `.md` 模板、`.py` 脚本、配置文件
- **Skill 与 settings.json 联动**：用 hooks 加自动行为；用 permissions 让 Skill 需要的命令自动放行

### 动手任务
- [ ] **第一个 Skill：`rag-project-build`**
  - 路径：`~/.claude/skills/rag-project-build/SKILL.md`
  - 功能：一键 `python -m rag.cli build`（先激活 conda env）
  - description 至少 2 句，覆盖典型触发场景
  - 测试：新会话里说「重建一下 rag 项目的索引」，看是否自动触发
- [ ] **第二个 Skill：`new-rag-project`**
  - 按 [memory/rag-project-layout](file:///C:/Users/ril/.claude/projects/f--source-code-direction-rag/memory/rag_project_layout.md) 创建新 RAG 子项目目录
- [ ] **改进一个已有 Skill**：fork 内置 `/verify` 到 `~/.claude/skills/verify-rag/`，加 RAG 项目专门的检验步骤
- [ ] 用 `update-config` skill 把上述 Skill 用到的命令加进 `settings.json` 的 `permissions.allow`

### 配套 notebook
✅ **[34_first_skill.ipynb](practice/stage2_进阶/34_first_skill.ipynb)** — 5 段模板 / 反例 Skill / checklist / 附属资源
✅ **[35_description_recall_test.ipynb](practice/stage2_进阶/35_description_recall_test.ipynb)** — 触发 / 不触发 10 对回归测试

### 推荐资源
- 本机：你自己写的 Skill 就是最好的参考
- [Anthropic · Skills 文档](https://docs.anthropic.com/) —— 搜 "skills"
- 社区：搜 GitHub `claude-code-skills` topic

### 能力检验
- [ ] 写一个 Skill，让另一个不熟悉你项目的人，仅靠看 SKILL.md 就能复现你的工作流
- [ ] description 能在 3 个不同表述的请求里都被准确触发，且不被无关请求误触发
- [ ] 解释「为什么 Skill 比把同样指令写进 CLAUDE.md 更好」（粒度 / 按需加载 / 可组合 / 可分享）

---

## 阶段 3 · 高级：Skill + sub-agent + MCP 组合（1 周）

### 核心知识点
- **Skill + sub-agent**：Skill 指示 Claude **spawn 一个 sub-agent** 处理大查询（隔离上下文 / 并行）
- **Skill + MCP**：Skill 描「做什么 + 步骤」，MCP 提供「具体工具」（如内部 API）
- **Skill 组合（meta-skill）**：一个 Skill 调用另一个 Skill —— 形成「个人工作流目录」
- **Skill 测试**：一组「触发示例 + 不该触发示例」做 description 召回回归

### 动手任务
- [ ] **组合 Skill：`rag-improve`** —— 步骤：
  1. spawn Explore sub-agent 找出 rag_project 当前瓶颈
  2. 根据结果调用 `/code-review` 给修改建议
  3. 落地后用 `/verify` 跑一遍
- [ ] **接入 MCP server**：开源 [filesystem MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) 或 [git MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/git)，在 Skill 里调它的工具
- [ ] 给 Skill 写一组**测试输入**（10 条），run 一遍看召回准确率

### 配套 notebook
✅ **[36_skill_subagent_compose.ipynb](practice/stage3_高级/36_skill_subagent_compose.ipynb)** — Skill 触发 sub-agent / meta-skill
✅ **[37_skill_hooks_permissions.ipynb](practice/stage3_高级/37_skill_hooks_permissions.ipynb)** — hooks 自动行为 + permissions 自动放行

### 推荐资源
- [MCP 官方 servers](https://github.com/modelcontextprotocol/servers)
- [Claude Code sub-agent 设计](https://docs.anthropic.com/) —— Agent SDK 部分

### 能力检验
- [ ] 写出一个 Skill，能在不影响主对话上下文的前提下 spawn sub-agent 完成耗 context 的工作
- [ ] 能讲清「Skill / sub-agent / MCP / settings hooks」各自管什么、不该管什么

---

## 阶段 4 · 专家：团队 Skill 库 / 版本化 / 发布（持续）

### 核心知识点
- **Skill 库分层**：个人（`~/.claude/skills/`）/ 项目（`<repo>/.claude/skills/`，跟代码走）/ 团队（共享 git 仓 / plugin）
- **版本管理**：把 Skill 仓当成代码仓维护（changelog、测试）
- **plugin 系统**：把一组相关 Skill 打成 plugin 分发
- **可观测性**：记录每个 Skill 被触发的频率、成功率、用户反馈
- **持续优化**：根据失败 case 调 description 与步骤

### 动手任务
- [ ] **建立 Skill 仓库**：单独 git 仓，结构 `skills/<name>/SKILL.md`，README 列出所有 Skill 与触发样例
- [ ] **写一组 RAG 专用 Skill** 至少 5 个：
  - `rag-bench` —— 跑评估集
  - `rag-add-doc` —— 增量加新文档
  - `rag-debug-retrieval` —— 给定 query 看为什么召不回
  - `rag-switch-embedding` —— 一键切换嵌入模型并重建
  - `rag-deploy-fastapi` —— 包成 API 服务并启动
- [ ] **把项目 Skill 放进 [rag_project/.claude/skills/](../../rag_project/)**（跟着 git 走），让任何 clone 该仓的人都能直接用
- [ ] 写一份「**Skill 设计 checklist**」：单一职责 / description 清晰 / 有反例 / 有恢复路径 / 破坏性操作要确认

### 配套 notebook
✅ **[38_skill_repo_skeleton.ipynb](practice/stage4_专家/38_skill_repo_skeleton.ipynb)** — Skill 仓 scaffolding / 项目级覆盖 / 版本化

### 推荐资源
- [Anthropic · Plugin 文档](https://docs.anthropic.com/) —— 搜 plugin / marketplace
- 业界参考：开源 Claude Code skill 集合

### 能力检验
- [ ] 你的 Skill 库被另一个人用了，并给出正面反馈
- [ ] 1 小时内为新项目写 3 个有用的 Skill
- [ ] 能维护一份 `skills/CHANGELOG.md`，记录 description 改动与原因

---

## 与其它章节的关系

- **强耦合**：[02-Agent](../02-Agent/) —— Skill 是 Agent 工作流的封装；理解 Agent 思想才能写好 Skill
- **下游被引**：
  - [05-评估](../05-评估与可观测性-Evaluation/) 可写 Skill 自动跑 eval
  - [06-部署](../06-工程化与部署-Deployment/) 可写 Skill 一键部署
  - [07-Capstone 项目 2](../07-综合项目-Capstone/) 把研究 Agent 沉淀为 `/research-topic` Skill
- **上游依赖**：[00-基础](../00-基础-Foundations/) 的 git / asyncio 基础

---

## 反模式

- ❌ **超大杂烩 Skill**：一个 Skill 想干 10 件事，description 写得很泛 → 召回乱七八糟
- ❌ **不写「不要在何时使用」**：被错误场景触发，污染主对话
- ❌ **硬编码绝对路径**：Skill 不可移植，换台机器全废
- ❌ **没有恢复路径**：Skill 中断后留下脏状态，下次跑就跑不通
- ❌ **用 Skill 替代 CLAUDE.md**：通用约束放 CLAUDE.md，特定流程才放 Skill
- ❌ **把秘密塞进 Skill**：Skill 可能被分享，token / 密码必须留在 `.env` 与 `settings.local.json`

---

## 前沿追踪

- 关注 `claude-code` GitHub topic 与 issue 区
- 关注 Anthropic 官方对 Skill / Plugin / Agent SDK 的版本更新
- 实践规律：**每解决一个重复任务 3 次以上，就考虑沉淀成 Skill**
