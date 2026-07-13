# 04-模型微调-Finetuning · Practice Lab

> 配套 [04-模型微调-Finetuning/README.md](../README.md) 的 Stage 1–4 动手实验代码。
> 环境 = `ft` conda env（**注意**：不能用 rag env，transformers 版本冲突）。

---

## 环境前置

```bash
# 创建 ft env（一次性）
conda create -n ft python=3.11 -y
conda activate ft
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes trl wandb sentencepiece einops

# 本地运行 jupyter
cd "f:/source/code/direction/rag/learning-roadmap/04-模型微调-Finetuning/practice"
jupyter lab
# 启动后选 "Python (ft)" kernel
```

**注意**：本机 `rag` env 的 `transformers 4.38.2` 太旧，训练会报错 → **必须新建 `ft` env**。

---

## 本机环境（实测）

- `ft` env 需要手动创建（参考上方）\n",
- 本 notebook OFFLINE 模式：所有 LLM 调用用 hash stub，训练用模拟\n",
- 真实训练需要安装 `transformers / datasets / accelerate / peft / trl / bitsandbytes`\n",
- 长上下文扩展需要 `YaRN / LongRoPE`\n",
- 分布式训练需要 `DeepSpeed`（ZeRO-1/2/3）或 `accelerate`（FSDP）

---

## 学习顺序

每个 notebook 结构：**学习目标 → 阶段拆解 → 深入思考 → 自检 ✅ → 下一步**。

### Stage 1 · 入门（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 39 | [stage1_入门/39_hf_ecosystem.ipynb](stage1_入门/39_hf_ecosystem.ipynb) | HF 三件套 / chat template / generate 参数 / 量化认知 | 60 min |

### Stage 2 · 进阶（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 40 | [stage2_进阶/40_lora_qlora_sft.ipynb](stage2_进阶/40_lora_qlora_sft.ipynb) | LoRA/QLoRA 参数 / 最小 SFT 脚本 / r=4/8/16/32 对比 | 90 min |

### Stage 3 · 高级（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 41 | [stage3_高级/41_dpo_preference_optimization.ipynb](stage3_高级/41_dpo_preference_optimization.ipynb) | DPO 原理 / 偏好数据合成 / C-Eval/CMMLU 对比 | 90 min |

### Stage 4 · 专家（1 个 notebook）

| # | 文件 | 主题 | 预计时长 |
|---|------|------|---------|
| 42 | [stage4_专家/42_distributed_longcontext_inference.ipynb](stage4_专家/42_distributed_longcontext_inference.ipynb) | DeepSpeed ZeRO-1/2/3 / YaRN 长上下文 / mergekit / 源码精读 | 90 min |

---

## 学完之后

回到 [04-模型微调-Finetuning/README.md](../README.md) 打 checkbox + 做能力检验。Stage 1–4 全部通过 → 可进 [05-评估](../05-评估与可观测性-Evaluation/) 或 [07-Capstone 项目 3](../07-综合项目-Capstone/)。
