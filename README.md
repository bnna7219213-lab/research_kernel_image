# T2Image Phase-A — 自研文生图研究内核

> 企业研发 · 8×A100-80GB · K8s（与 research_kernel_music 分时共享）
> 目标：PoC 产出 512×512 text-to-image 首版，可商用 + LoRA 可控微调

## 目录结构

```
.
├── configs/                     # 训练配置
│   ├── pixart_sigma.yaml
│   └── flux1_dev.yaml
├── docker/
│   ├── Dockerfile               # PyTorch 2.5 + CUDA 12.4 + FID/CLAP/PickScore/HPS
│   └── entrypoint.sh            # 统一 CLI：train/infer/eval/data
├── k8s/
│   └── submit_train.yaml         # 8 卡 Job + 4 卡 fallback + PriorityClass
├── scripts/
│   ├── download_datasets.sh      # LAION-Aesthetics (img2dataset) / COCO
│   ├── compute_data_fingerprint.py
│   └── clean_data.py             # CLIP / NSFW / dedup / recaption prep
├── src/
│   ├── train/                   # main + train_pixart + train_flux + infer_*
│   ├── eval/                    # FID / CLIP / PickScore / HPSv2.1
│   └── utils/                   # distributed / logger / seed
├── tests/
│   └── smoke_test.py            # 5 个冒烟
├── plan.md                      # 技术规划
├── roadmap_v0.1.md              # 路线图
└── .gitignore
```

## 快速开始

### 1. 本机冒烟（CPU 即可）

```bash
pip install pyyaml torch torchvision Pillow
python tests/smoke_test.py
```

### 2. K8s 训练（8×A100-80GB，与音乐项目分时段共享）

```bash
docker build -t your-registry.io/t2image:phase-a -f docker/Dockerfile .
docker push your-registry.io/t2image:phase-a

kubectl apply -f k8s/submit_train.yaml
kubectl logs -f job/t2image-train-pixart -n <namespace>
```

### 3. 数据管线

```bash
T2IMAGE_DATA_ROOT=/data/t2image bash scripts/download_datasets.sh laion --max 10000
python scripts/clean_data.py \
    --input_dir  /data/t2image/raw/laion_aesthetics \
    --output_dir /data/t2image/clean/laion_aesthetics \
    --clip_threshold 0.30 --min_short_side 256 --max_files 1000
python scripts/compute_data_fingerprint.py \
    --image_dir /data/t2image/clean/laion_aesthetics \
    --caption_csv /data/t2image/clean/laion_aesthetics/metadata.csv \
    --out /data/t2image/_spec/laion_aesthetics.fingerprint.json
```

### 4. 评估

```bash
docker run --gpus=1 -v /data:/data t2image:phase-a eval-fid \
    --gen_dir /data/t2image/outputs/pixart \
    --ref_dir /data/t2image/coco/val2014 \
    --out /data/t2image/eval/fid.json

docker run --gpus=1 -v /data:/data t2image:phase-a eval-clip \
    --image_dir /data/t2image/outputs/pixart \
    --captions_json /data/t2image/eval/demo_captions.json \
    --out /data/t2image/eval/clip.json

docker run --gpus=1 -v /data:/data t2image:phase-a eval-pick \
    --image_dir /data/t2image/outputs/pixart \
    --captions_json /data/t2image/eval/demo_captions.json \
    --out /data/t2image/eval/pick.json

docker run --gpus=1 -v /data:/data t2image:phase-a eval-hps \
    --image_dir /data/t2image/outputs/pixart \
    --captions_json /data/t2image/eval/demo_captions.json \
    --out /data/t2image/eval/hps.json
```

## 算力共享（与 T2Music 同池）

| 时间段 | Music 配额 | T2Image 配额 |
|---|---|---|
| W1–W8  | 75%        | 25%         |
| W9–W16 | 50%        | 50%         |
| W17–W24| 25%        | 75%         |

```yaml
# PriorityClass — music 高优先级 vs. image 普通优先级
kubectl apply -f k8s/priority_classes.yaml   # 见本仓库根 priority.yaml 注释
```

## 自动评估体系（每 2500 step）

| 指标 | 方向 | 目标（PoC 3B） | 开源基线 |
|---|---|---|---|
| FID-10K | ↓ | ≤ 18 | PIXART: ~9 , FLUX: ~5 |
| CLIP Score | ↑ | ≥ 0.32 | PIXART: 0.33 , FLUX: 0.35 |
| PickScore | ↑ | ≥ 20 | FLUX: ~21.5 |
| HPSv2.1 | ↑ | ≥ 30 | FLUX: ~32 |
| 盲测 MOS | ↑ | ≥ 3.5 / 5 | — |

## 许可

- 模型权重：PixArt-Σ (Adobe 自定义) / FLUX.1-dev (Apache-2.0)
- 数据：LAION-Aesthetics (研究) + CC-BY 自训
- 代码：企业自研 + 上游 Apache/MIT 复用
- 参见 `plan.md` §2 与 §9

## 下一步

- M2 渲染器训练：8 卡 × 6 周，50k step
- M3 DMD 蒸馏：目标 2-step 推理 ≤ 4s/张
- M4 垂类：电商图 / 海报 / IP 形象 LoRA

更多细节见 `plan.md`（技术规划）与 `roadmap_v0.1.md`（路线图）。
