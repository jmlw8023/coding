# 03-Claude-Skills · Practice Lab

> 配套 [03-Claude-Skills/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `rag` conda env（同 [00-基础/practice](../../00-基础-Foundations/practice/README.md) 的 kernel）。

---

## 环境前置

```bash
conda activate rag
cd "f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice"
jupyter lab
# 启动后选 "Python (rag)" kernel
```

`~/.claude/skills/` 目录**默认不存在**。notebook 会从零开始建。

---

## 本机环境（实测）

- `~/.claude/skills/` 不存在 → 第一节会**建这个目录**
- 沙箱化在 `_skill_sandbox/` 目录里跑，**不污染真实 `~/.claude/`**
- OFFLINE / ONLINE 都用规则模拟（不依赖 Ollama），所有示例 skill 都是纯 Python 工具

---

## 学习顺序

每个 notebook 结构：**学习目标 → 阶段拆解 → 深入思考 → 自检 ✅ → 下一步**。

### Stage 1 · 入门（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 33 | [stage1_入门/33_skill_browser.ipynb](stage1_入门/33_skill_browser.ipynb) | 拆 Claude Code 自带 Skill + Skill 存放结构 + 沙箱建模 | 60 min |

### Stage 2 · 进阶（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 34 | [stage2_进阶/34_first_skill.ipynb](stage2_进阶/34_first_skill.ipynb) | 写第一个 Skill：SKILL.md frontmatter + 步骤 + 示例 + 反例 | 75 min |
| 35 | [stage2_进阶/35_description_recall_test.ipynb](stage2_进阶/35_description_recall_test.ipynb) | description 召回回归测试：10 触发 / 10 不该触发 | 75 min |

### Stage 3 · 高级（2 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 36 | [stage3_高级/36_skill_subagent_compose.ipynb](stage3_高级/36_skill_subagent_compose.ipynb) | Skill 触发 sub-agent + meta-Skill 组合 | 75 min |
| 37 | [stage3_高级/37_skill_hooks_permissions.ipynb](stage3_高级/37_skill_hooks_permissions.ipynb) | hooks 自动行为 + permissions 自动放行 | 60 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 38 | [stage4_专家/38_skill_repo_skeleton.ipynb](stage4_专家/38_skill_repo_skeleton.ipynb) | Skill 仓库 scaffolding + 复用为项目级别 Skill + 多级加载 | 90 min |

---

## 学完之后

回到 [03-Claude-Skills/README.md](../README.md) 打 checkbox + 做能力检验。Stage 1–4 全部通过 → 可进 [04-微调](../../04-模型微调-Finetuning/) 或 [07-Capstone 项目 2](../../07-综合项目-Capstone/)。
