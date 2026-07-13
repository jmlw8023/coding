# My Skill Repo

> 一组 Claude Code Skill 集合。

## 全部 Skill

| Skill | 用途 | 版本 |
|-------|------|------|
| [commit-pr](skills/commit-pr/) | 提交 + 推 PR | 0.1.0 |
| [audit-rag](skills/audit-rag/) | 审计 rag_project 配置 | 0.1.0 |
| [study-topic](skills/study-topic/) | 深入研究 + 自动存笔记 | 0.1.0 |

## 装
```bash
./install.sh    # 复制到 ~/.claude/skills/
```

## 维护约定

- 每个 Skill 一个子目录
- `description` 改了必须改 `CHANGELOG.md`
- 每周跑一遍 `tests/recall_test.py`（如果存在）
