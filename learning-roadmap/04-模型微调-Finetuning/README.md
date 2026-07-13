# 04 - 模型微调 Fine-tuning

> **一句话定位**：当 Prompt + RAG 解决不了的问题（行为风格 / 大量私有知识内化）出现时，用 LoRA / QLoRA / DPO 训出领域模型。
> **时间预算**：4–12 周（**最贵的一章**，硬件与时间投入最大）。
> **适用判断**：能独立交付一个领域模型（数据 → 训 → 评 → 部署）、看新微调论文能立刻判断「是否值得复现」 → 进 [06-部署](../06-工程化与部署-Deployment/) 或 [07-Capstone 项目 3](../07-综合项目-Capstone/)。

---

## 学习目标

完成本章后，你应当：
1. 判断「我该用 RAG 还是该微调」—— 给一个业务问题能 3 分钟给出推荐
2. 在 16GB 单卡上跑通 QLoRA 7B 微调 + 评估 baseline
3. 能讲清 SFT vs DPO 各自解决什么 / 不能解决什么
4. 能为领域设计「不会作弊」的评估集（含正例、反例、边界例）
5. 一周内独立交付领域模型（数据 → 训 → 评 → 量化 → vLLM 部署）

---

## 实战状态

⏳ **notebook 待建**（路线已写）。本章需新建 `ft` env，**不能直接用 `rag` env**（transformers 版本冲突，详见 [memory/rag-conda-env](file:///C:/Users/ril/.claude/projects/f--source-code-direction-rag/memory/rag_conda_env.md)）。

---

## 起步前的现实检查

| 问题 | 答案 |
|------|------|
| 我应不应该微调？ | 90% 的场景应先用 Prompt + RAG，调不上去再考虑微调 |
| 我的机器够用吗？ | LoRA 7B Q4 约 16GB 显存；QLoRA 4-bit 约 8GB。Windows + 单卡也能起步 |
| 我的 `rag` env 能用吗？ | **不能**。`transformers 4.38.2` 太旧。**必须新建 `ft` env** |
| 数据从哪来？ | 数据是 80% 的工作量。没准备好数据就别开始 |

### 新建 `ft` env（一次性）

```bash
# 不要污染 rag env
conda create -n ft python=3.11 -y
conda activate ft

# 核心
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 看你 CUDA 版本
pip install transformers datasets accelerate peft bitsandbytes trl
pip install wandb                                                       # 可选，做实验记录

# 中文场景常用
pip install sentencepiece einops
```

> 网络不稳定就用清华或阿里源；torch 安装最容易失败，必要时下 wheel 装。

---

## 全章地图

```
                       决策点
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
    Prompt 调好了吗？─────否────► 调 Prompt
       │ 是
       ▼
    RAG 接进来够吗？──────否────► 用 01 章
       │ 否
       ▼
    需要改「行为风格」或「内化大量私有知识」？
       │
       ▼
    ┌────────────────────────────────┐
    │   开始考虑微调                   │
    └─────────────┬──────────────────┘
                  ▼
    ┌────────────────────────────────┐
    │ 1. 准备数据                     │
    │ 2. 选基座（开源 vs 闭源）         │
    │ 3. 选方法（LoRA/QLoRA/Full）     │
    │ 4. 训练 → 评估 → 迭代            │
    │ 5. 部署（量化/推理优化）          │
    └────────────────────────────────┘
```

---

## 阶段 1 · 入门：HuggingFace 生态 + 跑通推理（1–2 周）

### 核心知识点
- `transformers` 三件套：`AutoModel`、`AutoTokenizer`、`pipeline`
- `datasets` 库：本地 JSON / CSV / Parquet 加载、`map` / `filter`、流式
- 模型家族认知：Llama / Qwen / Gemma / Mistral / DeepSeek 各自定位
- 推理基础：`model.generate()` 各参数（max_new_tokens / temperature / do_sample）
- 量化基础概念：fp16 / bf16 / int8 / int4 / GGUF / GPTQ / AWQ

### 动手任务
- [ ] 本机 ft env 跑通 Qwen 推理（`Qwen/Qwen2.5-0.5B-Instruct`），手撸 `apply_chat_template` → `generate`
- [ ] 用 `datasets` 加载本机 JSON（造 10 条 `{"prompt", "completion"}`），完成 `map` 改写成 chat 模板格式
- [ ] 同一段中文 prompt 跑 Qwen / Llama3 / Gemma 三家，对比效果写 200 字结论
- [ ] 在本机 Ollama 里 pull 一个 GGUF 模型，理解 GGUF vs HF safetensors 的差异

### 配套 notebook
⏳ 待建（计划：HF 三件套 / chat template / generate 参数 / GGUF 对比）

### 推荐资源
- [HuggingFace Course (free)](https://huggingface.co/learn/llm-course/chapter1/1) —— Chapter 1–7
- [Transformers Quicktour](https://huggingface.co/docs/transformers/quicktour)
- [Qwen 官方仓库](https://github.com/QwenLM/Qwen2.5) —— 中文模型首选

### 能力检验
- [ ] 不看文档，10 分钟内写完「加载模型 → 应用 chat template → 生成 → 解码」全流程
- [ ] 说出 fp16 / bf16 / int4 各自精度损失与显存收益
- [ ] 解释「同样是 7B 模型，为什么 GGUF 跑得比 safetensors 快」

---

## 阶段 2 · 进阶：LoRA / QLoRA 跑通第一次微调（1–2 周）

### 核心知识点
- **PEFT 全景**：Full FT / LoRA / QLoRA / Prefix Tuning / DoRA / LoRA+
- **LoRA 关键超参**：`r`（4/8/16/32）、`alpha`（通常 `= 2·r`）、`target_modules`、`dropout`
- **SFT 数据格式**：多轮对话 vs 单轮指令；**必须用 `tokenizer.apply_chat_template`**，否则效果腰斩
- **训练监控**：`loss` 但更要看 `eval_loss`；显存峰值 + `gradient_accumulation_steps`；wandb / tensorboard

### 动手任务
- [ ] **第一个 SFT (TRL)**：
  - 基座：`Qwen/Qwen2.5-0.5B`（最小代价跑通）
  - 数据：100 条「特定风格回答」样本（如总用「老兄你听好了」开头）
  - 方法：LoRA r=8
  - 看：调完后模型能不能学会这个风格
- [ ] **QLoRA 跑 7B**：基座 `Qwen/Qwen2.5-7B-Instruct`、500+ 领域 QA、QLoRA 4-bit + LoRA r=16，单卡 16GB 应能跑通
- [ ] **对比实验**：r=4 / 8 / 16 / 32 各跑一次，画 eval_loss 曲线
- [ ] **合并 + 量化**：LoRA 权重 merge 回基座，导出为完整模型，再转 GGUF 让 Ollama 能跑

### 配套 notebook
⏳ 待建（计划：TRL SFT / QLoRA 7B / r 对比 / merge + GGUF）

### 推荐资源
- [HF PEFT 文档](https://huggingface.co/docs/peft)
- [HF TRL 文档](https://huggingface.co/docs/trl) —— SFTTrainer / DPOTrainer 都在这
- [LoRA 论文 (2021)](https://arxiv.org/abs/2106.09685) / [QLoRA 论文 (2023)](https://arxiv.org/abs/2305.14314)
- [unsloth](https://github.com/unslothai/unsloth) —— LoRA/QLoRA 训练加速 2x，Windows 友好
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) —— 国人主导的全流程工具

### 能力检验
- [ ] 不查文档，30 分钟内写完最小 SFT 训练脚本（数据 / 模型 / LoRA / Trainer / save）
- [ ] 给一份 loss 不下降的训练日志，能列出 5 个排查方向（数据格式 / lr / target_modules / chat template / dtype）
- [ ] 解释「为什么 QLoRA 比 LoRA 显存省、精度损失为啥可接受」

---

## 阶段 3 · 高级：DPO / RLAIF / 数据合成 / 评估（2–4 周）

### 核心知识点
- **SFT 之后的偏好优化**：RLHF（经典三步走）/ **DPO（实战首选，不训 reward model）** / IPO / KTO / ORPO / RLAIF
- **偏好数据格式**：`{"prompt", "chosen", "rejected"}`；1000–10000 条高质量对就能看到效果
- **数据合成**：用强模型（Claude / GPT-4 / DeepSeek）生成训练样本；[Distilabel](https://github.com/argilla-io/distilabel) 流水线；**人工抽检不可省**
- **评估**：
  - 基础能力没掉：MMLU / CMMLU / C-Eval
  - 目标能力提升：自建领域 eval set
  - 风格 / 安全 / 拒答：单独 eval set
  - 与 [05 章](../05-评估与可观测性-Evaluation/) 深度交叉
- **何时考虑连续预训练（CPT）**：领域词汇/概念陌生 + 数亿 token+ 数据 → 一般在 SFT 之前做

### 动手任务
- [ ] **跑通 DPO**：用上阶段 SFT 模型作 base + 300 条偏好对 + TRL `DPOTrainer`，评估风格符合度 / 拒答率 / 通用能力是否退化
- [ ] **数据合成实战**：给定领域，用 Claude / DeepSeek API 自动生成 1000 条 `(question, answer)`，人工抽检 5%，跑 SFT 看效果
- [ ] **跑 C-Eval / CMMLU**，看微调有没有损坏通用能力
- [ ] **失败案例分析**：按「数据质量 / 数据量 / 超参 / 评估方法」四象限定位

### 配套 notebook
⏳ 待建（计划：DPO / 数据合成 / C-Eval / 失败 4 象限）

### 推荐资源
- [DPO 论文 (2023)](https://arxiv.org/abs/2305.18290)
- [HF DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Argilla Distilabel](https://github.com/argilla-io/distilabel)
- [C-Eval](https://cevalbenchmark.com/) / [CMMLU](https://github.com/haonan-li/CMMLU)
- [OpenCompass](https://github.com/open-compass/opencompass)

### 能力检验
- [ ] 讲清 SFT vs DPO 各自适合什么 / 不能解决什么
- [ ] 用 DPO 把模型的「拒答率」「输出风格」至少一项打到目标值
- [ ] 为领域设计「不会作弊」的评估集（正例 + 反例 + 边界例）

---

## 阶段 4 · 专家：多卡、长上下文、CPT、推理优化（持续）

### 核心知识点
- **分布式训练**：DataParallel vs DDP vs DeepSpeed ZeRO 1/2/3 vs FSDP；多机多卡 `accelerate` / `torchrun` / `deepspeed`；NVLink / PCIe / 千兆 vs 万兆
- **长上下文训练**：YaRN / LongRoPE / NTK scaling；FlashAttention / RingAttention；序列并行；长样本数据混合
- **连续预训练 (CPT)**：学习率远小于 from-scratch（1e-5 量级）；领域 + 通用混合（防灾难性遗忘）；监控 PPL + 下游评估
- **推理优化**（与 [06 章](../06-工程化与部署-Deployment/) 交叉）：GPTQ / AWQ / GGUF / fp8；vLLM / TGI / TensorRT-LLM / llama.cpp；speculative decoding / continuous batching / PagedAttention
- **模型合并 / 适配器堆叠**：多个 LoRA 合到一个 base；Mixture of LoRAs；[mergekit](https://github.com/cg123/mergekit)

### 动手任务
- [ ] **多卡微调**（如有条件）：DeepSpeed ZeRO-2 或 FSDP 跑 7B 全量微调，对比单卡 LoRA / 单卡 QLoRA / 多卡 Full 的吞吐与质量
- [ ] **扩展上下文**：8k 模型用 YaRN 扩到 32k，做长文 needle-in-haystack 测试
- [ ] **完整链路项目**：合成 + 真实数据，对 7B 完成 SFT → DPO → 评估 → 量化 → vLLM 部署
- [ ] **读源码**：精读 `transformers/models/llama/modeling_llama.py` + `peft` 源码，讲清 LoRA 怎么 hook 进去的
- [ ] 写一份「领域大模型构建白皮书」

### 配套 notebook
⏳ 待建（计划：DeepSpeed / YaRN / mergekit 实操）

### 推荐资源
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [HF Accelerate 文档](https://huggingface.co/docs/accelerate)
- [vLLM 文档](https://docs.vllm.ai/) / [llama.cpp](https://github.com/ggerganov/llama.cpp)
- 论文：[YaRN (2023)](https://arxiv.org/abs/2309.00071)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[vLLM/PagedAttention](https://arxiv.org/abs/2309.06180)
- [HF Alignment Handbook](https://github.com/huggingface/alignment-handbook) —— 端到端代码参考

### 能力检验
- [ ] 一周内独立交付一个领域模型（数据 → 训 → 评 → 部署），含完整实验记录
- [ ] 看一篇新微调论文能立刻判断「是否值得复现」
- [ ] 能给团队做技术决策：「这件事该 RAG / 该微调 / 该 fine-tune embedding」

---

## 与其它章节的关系

- **强耦合**：
  - [01-RAG](../01-RAG/) —— 信息更新快 / 数据可外置 → RAG；风格 / 拒答 / 模型本身能力短板 → 微调；**大多场景两者都要**
  - [05-评估](../05-评估与可观测性-Evaluation/) —— 没有评估的微调是赌博；DPO 偏好数据就来自评估失败 case
- **下游被引**：
  - [02-Agent](../02-Agent/) 可以训出更可靠的 function-calling 行为
  - [06-部署](../06-工程化与部署-Deployment/) —— 训完不部署等于没训
  - [07-Capstone 项目 3](../07-综合项目-Capstone/) 是本章的综合演练
- **上游依赖**：[00-基础](../00-基础-Foundations/) stage 3-4 的训练稳定性 + 精度知识

---

## 反模式

- ❌ **没评估集就开始训**：你不知道有没有变好
- ❌ **数据 < 100 条 + 想全量微调**：必定过拟合且通用能力崩
- ❌ **不用 chat template**：模型不知道你在跟它对话，loss 看起来在降但什么都没学到
- ❌ **直接调最大模型**：先在 0.5B / 1.5B 跑通流程，再上 7B / 14B
- ❌ **盲信 wandb 曲线**：loss 降 ≠ 模型变好，必须看 eval + 人工抽检
- ❌ **微调代替 RAG**：用微调注入「明天就会过期」的事实知识 —— 最贵最低效
- ❌ **只跑别人脚本不读论文**：到不了高级阶段

---

## 前沿追踪

- arXiv：每天扫 cs.CL 与 cs.LG 的「LoRA / DPO / efficient fine-tuning」关键词
- 关键团队：Hugging Face、EleutherAI、Stanford CRFM、上海 AI Lab、Qwen 团队、DeepSeek 团队
- 关键开源：[HF Alignment Handbook](https://github.com/huggingface/alignment-handbook)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[unsloth](https://github.com/unslothai/unsloth)、[axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- 公开榜单：Open LLM Leaderboard、OpenCompass、Chatbot Arena
