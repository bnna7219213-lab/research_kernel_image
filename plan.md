# 自研文生图大模型技术规划 — T2Image

> 版本：v0.1（2026-09-22）
> 关联项目：`research_kernel_music`（共享 8×A100-80GB K8s 池，分时时段调度）
> 状态：规划阶段，目标 2026.Q4 完成首版 PoC（512px text-to-image）

---

## 1. 背景与目标

### 1.1 为什么要自研

文生图（Text-to-Image）是多模态大模型最成熟的分支之一，在创意、广告、电商、游戏、教育等场景需求明确。主流 SOTA（Stable Diffusion 3.5、Flux 2、PixArt-Σ、Kolors 3）已开放商用许可（Apache / MIT），自研的策略在于：

- **可控性**：完全拥有数据、模型权重、超参 → 可针对垂类（电商图、海报、IP 形象）做深度定制
- **成本**：大批量推理时，自托管模型单价远低于 API 调用（GPT-Image、Midjourney、即梦）
- **合规**：自有数据 + 自有权重 → 规避第三方 API 内容政策、隐私、版权条款风险
- **迭代速度**：自研模型可在 1-2 周内响应业务需求，不受第三方发布节奏限制

### 1.2 自研目标（首版 PoC，M0–M3）

| 维度 | 目标 |
|---|---|
| 输入 | 自然语言 prompt（中英文，≤200 token） |
| 输出 | 512×512 RGB 图片，≥4s 推理/帧（A100-80GB） |
| 训练数据 | ≥1000 万图文对齐样本（清洗后） |
| 模型规模 | 渲染器 3B+ VAE + text encoder（复用 FLAN-T5 / CLIP ViT-L） |
| 许可 | Apache / MIT 开源框架，确保商用 |
| 评估 | FID ≤ 15 / CLIP score ≥ 0.33 / PickScore ≥ 20.5 / HPSv2.1 ≥ 30 |

---

## 2. 技术格局（2026 年 9 月快照）

### 2.1 主流开源路线

| 模型 | 规模 | 核心架构 | 许可 | 备注 |
|---|---|---|---|---|
| **Stable Diffusion 3.5 Large** | 9B (DiT) | Flow Matching + MMDiT | Stability AI 商用 | 生产力首选，FP8 量化推理 |
| **Flux 2** | 12B (DiT) | Flow Matching + RoPE | Apache 2.0 | 社区生态最强，多 LoRA |
| **PixArt-Σ** | 0.6–6B | DiT + T5 + DMD 蒸馏 | Adobe 商用 | 训练快，支持高分辨 |
| **Kolors 3** | GLM-4 + SD3 | 中英双语对齐 | 智谱开源 | 中文解析最强 |
| **Sana** | 0.6B | Linear DiT + DCAE | Apache 2.0 | 高性价比高分辨率 |
| **Lumina-Next** | 2B | Flow Matching + DiT | Apache 2.0 | 快手开源 |

### 2.2 关键组件选型共识

```
┌──────────────────────────────────────────────────────┐
│  Text → [CLIP-T / FLAN-T5] → text_embed              │
│                                        ↓             │
│  Noise z ~ N(0,I) → [DiT / MMDiT] → latent z'       │
│                                        ↓             │
│  z' → [SD-VAE / DCAE] → 512×512 RGB                 │
│                                        ↓             │
│    [可选] ControlNet / LoRA / IP-Adapter 控制        │
└──────────────────────────────────────────────────────┘
```

**组件对比**：

| 组件 | 候选 | 推荐 | 依据 |
|---|---|---|---|
| VAE | SD-VAE / DCAE / Cosmos | DCAE（ACE 系同源） | 重建 FAD 优 |
| Text Encoder | CLIP ViT-L / FLAN-T5 / GLM | 双路（CLIP 语义 + FLAN-T5 空间推理） | MMDiT 风格 |
| DiT Base | 自研 / PixArt-Σ / SD3.5 | 自研（基于 PixArt-Σ 起步） | 许可友好 |
| 训练目标 | Flow Matching / DMD / REPA | Flow Matching + REPA 语义对齐 | 收敛稳 |
| 蒸馏 | VAE-aware DMD-2 / Consistency | DMD 2-step | 推理 4 step |
| 推理加速 | FP8 / INT8 / DeepCache | DeepCache + torch.compile | 工程成熟 |

---

## 3. 自研方案：T2Image-v0

**核心思路**：复用开源框架（PixArt-Σ + Flux 训练栈）建立 PoC，重点投入在**垂类数据**和**条件注入**，不在从零训基础模型（ROI 太低）。

### 3.1 架构选择

```
[FLAN-T5-XXL / CLIP ViT-L]  →  双 text_embed  (768+1024 dim)
           ↓  concat + cross-attention
[Noise_z + timestep_embed]  →  MMDiT DiT (6–12 blocks, dim=2048)
           ↓  输出
[DCAE Decoder]              →  8×8 latent → 512×512 RGB
```

**阶段演进**：
- **PoC（M0–M2）**：固定 FLAN-T5 text encoder，训一个 3B DiT，512px
- **拓展（M3–M4）**：扩展到 768/1024px，加入 panoramic / hi-res cascade
- **量产（M5+）**：ControlNet、LoRA 训练台、IP-Adapter、视频帧生成

### 3.2 训练策略

| 阶段 | 目标 | 数据 | 算力 | 算法 |
|---|---|---|---|---|
| **Stage A** | VAE 自训练 + text-aligned codec | 自家图文对 + 公开 | 8×A100×4w | SD-VAE 基线 + DCAE 对比 |
| **Stage B** | DiT 渲染器预训练 | 10M+ 图文对 | 8×A100×6w | Flow Matching + REPA |
| **Stage C** | DMD 蒸馏对齐 | Stage B 数据 | 8×A100×2w | 2-step 对齐 |
| **Stage D** | 垂类微调 + RLHF | 业务数据 | 按垂类规模 | LoRA + DPO |

---

## 4. 数据方案

### 4.1 公开数据集（PoC 阶段主要来源）

| 数据集 | 规模 | 许可 | 用途 |
|---|---|---|---|
| **LAION-Aesthetics V2** | 过滤子集 1B+ | 研究 | 主训练集 |
| **Recap-DataComp** | 1B+ 高质量 recaption | CC | 语义对齐 |
| **COCO / COCO-1B** | 训练标准 | CC | 评估对齐 |
| **JourneyDB / Midjourney-2B** | 生成-合成数据 | 研究 | 高审美数据 |
| **Coyo-700M** | 清洗版 LAION | 开源 | 备用扩充 |

### 4.2 数据清洗管线

1. **NSFW 过滤**：Safety clip + 视觉黑盒过滤
2. **图文相关度**：CLIP score ≥ 0.30（严格 ≥ 0.33）
3. **分辨率分级**：小图(128)/中图(256)/大图(512+)
4. **去重**：pHash + 语义去重
5. **美学评分**：CLIP美学预测器保留 top-50% 高美学子集

### 4.3 自有数据接入

- 电商商品图、UI 截图、广告素材、内部设计等
- 自动化 recaption：使用 BLIP3 / GPT-Image 重 caption，统一 prompt 格式
- 自有数据与公开数据**分层训练**，比例可动态调

---

## 5. 训练管线

### 5.1 训练超参（PoC 参考）

| 超参 | 值 |
|---|---|
| 分辨率 | 512×512 (latent 64×64×4) |
| VAE | f8c8 DCAE / SD-VAE f8 |
| 最大序列长度 | FLAN-T5 + CLIP 各 77 token |
| 批大小（全局） | 256–1024（grad accum 控制） |
| 最大训练 tokens | ~ 500B tokens |
| 优化器 | AdamW (lr=1e-4, β=(0.9,0.999), wd=0.01) |
| LR scheduler | cosine with 1k warmup |
| EMA | decay=0.9999 |
| 精度 | bf16 (param) / fp32 (optimizer states) |

### 5.2 分布式（8×A100-80GB）

- FSDP-2 full_shard（ZeRO-3）
- activation checkpointing on DiT blocks
- torch.compile 全模型
- 梯度累积控制全局 batch（≥256 per step）
- 优先保证 8 卡同时跑；如被音乐任务抢占 → 退到 4 卡

### 5.3 评估体系（每 5000 step）

- **FID-10K**：MS-COCO 验证集对照
- **CLIP Score**：图文相关度
- **PickScore / ImageReward**：人类偏好对齐
- **HPSv2.1**：Human Preference Score
- **生成质量盲测**：内部 10 人 × 50 prompts 打分

---

## 6. 算力估算（PoC 3B, 8×A100-80GB）

| 阶段 | 规模 | 估算 GPU-h |
|---|---|---|
| VAE 训练 | 50M 图片 | ~800 |
| DiT 预训练 | 3B DiT × 256 batch × 50k step | ~3000 |
| DMD 蒸馏 | 1B 数据 × 10 step | ~1200 |
| 垂类微调 | LoRA + 1w 步 | ~300 |
| **合计 PoC** | — | **~5300 GPU-h** ≈ 连续跑 4 周 |

> 与音乐大模型**分时共享** 8 卡池，每项目各有 50% 时间配额 → PoC 时间 ×2

---

## 7. 里程碑

| 里程碑 | 关联 | 交付物 |
|---|---|---|
| M0 规划评审 | 本文档 | plan.md + roadmap |
| M1 基线跑通 (W4) | PixArt-Σ + 数据 | 推理 demo + 训练日志 |
| M2 短图 PoC (W10) | Stage B 完成 | 512px 可用模型 |
| M3 蒸馏提速 (W14) | Stage C | 4-step 推理 |
| M4 首版评审 (W20) | Stage D 初版 | 垂类微调 demo |
| M5 产品化 | 接入 API 服务 | 生产上线 |

---

## 8. 工作量与分工（建议）

| 角色 | 人数 | 核心职责 |
|---|---|---|
| 数据工程师 | 1–2 | 数据管线、清洗、标注、recaption |
| 训练框架工程师 | 1–2 | 分布式、FSDP、训练模板 |
| 算法研究员 | 1–2 | 模型架构、损失设计、评估体系 |
| 评测 & 产品 | 1 | 盲听、prompt 工程、业务对接 |
| 项目经理 | 1 | 进度、联动音乐项目调度 |

---

## 9. 风险表

| 风险 | 等级 | 影响 | 缓解 |
|---|---|---|---|
| **算力与音乐项目冲突** | 高 | 单项目仅 4 卡，耗时翻倍 | 明确分时段策略，CI 化排队 |
| **数据版权与合规** | 高 | 自有 recaption 数据来源不清 | 法务前置，自有数据独立分区 |
| **评估与业务脱节** | 中 | FID 好看但业务体验差 | 双轨评估（自动 + 人工），业务方提前介入 |
| **迭代慢于开源 SOTA** | 中 | 刚训好 Flux/SD 新版本超越 | 不追 SOTA，重点定制垂类；保持 2 周内 LoRA 节奏 |
| **训练 loss spike / OOM** | 中 | 长时间训练中断 | checkpoint 频繁保存、float32 关键 state |
| **中文文本解析弱** | 低 | FLAN-T5 中文不如 Kolors | 可在 text encoder 层加双语对齐微调 |

---

## 10. 与 music 项目的协同

- **代码复用**：eval 体系框架、K8s 任务模板、W&B 追踪、数据指纹
- **算力共享**：分时池（音乐 vs 文生图各 50%，或按 2 周轮换）
- **经验转化**：音乐 codec 的 DCAE 设计可迁移到图像 VAE
- **联合产出**：音乐封面图生成 + 视频多模态（长期）
