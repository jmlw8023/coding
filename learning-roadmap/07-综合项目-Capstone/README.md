# 07-综合项目-Capstone · Practice Lab

> 配套 [07-综合项目-Capstone/README.md](../README.md) 的实战项目实践代码。
> **每个项目独立 git 仓，本路线只提供实践指导 notebook。**

---

## 环境前置

```bash
# 项目 1：企业知识库 RAG
cd "f:/source/code/direction/rag/learning-roadmap/07-综合项目-Capstone/practice/51_kb_rag_project.ipynb"
jupyter lab
# 启动后选 "Python (rag)" kernel

# 项目 2：研究Agent
cd "f:/source/code/direction/rag/learning-roadmap/07-综合项目-Capstone/practice/52_研究Agent项目.ipynb"
jupyter lab
# 启动后选 "Python (rag)" kernel

# 项目 3：领域微调+RAG 混合
cd "f:/source/code/direction/rag/learning-roadmap/07-综合项目-Capstone/practice/53_领域微调RAG项目.ipynb"
jupyter lab
# 启动后选 "Python (ft)" kernel（需要新建 ft env）

# 项目 4：多模态文档助手
cd "f:/source/code/direction/rag/learning-roadmap/07-综合项目-Capstone/practice/54_多模态文档助手项目.ipynb"
jupyter lab
# 启动后选 "Python (rag)" kernel
```

**注意**：项目 3 需要新建 `ft` env（见 04 章 README）。

---

## 学习顺序

每个 notebook 结构：**项目目标 + 预备 → 阶段拆分 → 技术实现细节 → 评估指标 → 部署与监控 → 深入思考 → 自检**。

### 项目 1：企业知识库 RAG 系统（RAG 工程方向）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 51 | [51_kb_rag_project.ipynb](51_kb_rag_project.ipynb) | 完整企业知识库项目（50+ 文档 / FastAPI / Docker / Langfuse） | 4-6 周 |

### 项目 2：研究 / 信息聚合 Agent（Agent 方向）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 52 | [52_研究Agent项目.ipynb](52_研究Agent项目.ipynb) | 研究Agent + 技能化（web_search / read_page / 长期记忆） | 2-4 周 |

### 项目 3：领域微调 + RAG 混合系统（模型方向）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 53 | [53_领域微调RAG项目.ipynb](53_领域微调RAG项目.ipynb) | QLoRA 7B + RAG 混合 + 3 个对照组 + 30 题 eval | 6-8 周 |

### 项目 4：多模态文档助手（扩展方向）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 54 | [54_多模态文档助手项目.ipynb](54_4500a934da34多模态文档助手项目.ipynb) | PDF/PPT/DOCX 图文混排 + 混合检索 + 答案定位 | 3-5 周 |

---

## 学完之后

1. **选 2-3 个项目做成 git 仓**
2. **每项目走完「数据 → 索引 / 训练 → 评估 → 部署 → 监控」全循环**
3. **写一篇技术博客 / 内部分享**
4. **整理成个人技术站点 / GitHub Profile**
5. **结束本路线，成为给别人写路线的人**

---

## 项目路线图

```
00-02 章节技能积累
         ↓
┌────────────────────────────────────┐
│ 项目 1：企业知识库 RAG（RAG 工程）     │ ── 路径 A 收官
│ 项目 2：研究Agent（Agent 方向）        │ ── 路径 B 收官
│ 项目 3：领域微调 + RAG（模型方向）     │ ── 路径 C 收官
│ 项目 4：多模态文档助手（扩展）        │
└────────────────────────────────────┘
         ↓
   个人技术站点 / GitHub Profile
```

---

## 注意事项

- **不要 4 个项目同时开**：你会一个都做不完
- **每次项目要跑完整循环**：数据 → 索引 / 训练 → 评估 → 部署 → 监控
- **代码要留住**：每项目独立 git 仓 + README，3 个月后回看不至于陌生
- **碰到瓶颈就回到对应章节**：本章不是孤立的，是 01-06 的试炼场
