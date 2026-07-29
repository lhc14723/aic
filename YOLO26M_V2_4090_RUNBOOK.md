# YOLO26m-V2 单卡 RTX 4090 运行手册

这份文档是当前比赛主路线的唯一推荐入口。旧的
`*_4090d_clean.yaml`、`train_fusion.yaml` 和 `supervisor_4090d_clean.sh`
属于上一版 YOLO26s/V1 实验，不要用于本次 YOLO26m-V2 训练。

## 1. 数据与仓库边界

代码可以上传到 GitHub。官方训练集、测试集、处理后的 TIFF、标签和测试预测不得上传到
公开仓库；项目的 `.gitignore` 已忽略 `data/`、`artifacts/` 和 `outputs/`。

在服务器上预期有以下目录：

```text
aic/
├── aic_mm/
├── configs/
├── scripts/
├── data/
│   └── processed/
│       ├── aic_multispectral.yaml
│       ├── aic_multispectral_all.yaml
│       ├── images/train/   # 1600 TIFF
│       ├── images/val/     # 400 TIFF
│       ├── images/test/    # 1000 TIFF
│       ├── labels/train/   # 1600 TXT
│       └── labels/val/     # 400 TXT
└── weights/
    └── yolo26m.pt
```

数据 YAML 已改为相对自身位置解析。完整复制 `data/processed` 后，不需要根据服务器目录
修改 `path`。

## 2. 创建环境

在服务器项目根目录执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n aic-mm python=3.10.20 pip -y
conda activate aic-mm

python -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements-aic-mm.txt
```

如果服务器的 Conda 安装位置不是 `/root/miniconda3`，只需要把第一条命令换成实际路径。

每次重新连接服务器都执行：

```bash
conda activate aic-mm
cd /你的数据盘路径/aic
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
```

## 3. 准备公开预训练权重

如果仓库中没有 `weights/yolo26m.pt`，在项目根目录执行：

```bash
mkdir -p weights
python -c "from ultralytics.utils.downloads import attempt_download_asset; print(attempt_download_asset('weights/yolo26m.pt'))"
```

该文件约 43MB。不要让程序在找不到 m 权重时退回 s 权重。

## 4. 上线前强制检查

```bash
python -m scripts.preflight_4090
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

必须看到：

```text
"status": "READY"
```

以及全部测试通过。预检会核对 CUDA、GPU 显存、数据数量、8通道、12类别、YOLO26m、
V2融合和各阶段 batch/分辨率。任何一项失败都不要开始正式训练。

## 5. 960分辨率冒烟

这一步使用5%训练数据跑1轮，但保持正式第一阶段的 `batch=8`、增强和多尺度设置：

```bash
python -m scripts.train --config configs/smoke_fusion_v2_m_4090.yaml --yes
```

只检查 CUDA、显存、训练、验证和权重保存是否正常。输出目录含
`do_not_submit`，其权重不得提交。

如果冒烟发生 CUDA OOM，把以下两个配置的 `batch` 同时从8改为6，再重新冒烟：

```text
configs/smoke_fusion_v2_m_4090.yaml
configs/train_fusion_v2_m_4090.yaml
```

不要先降低 `imgsz`，也不要关闭 AMP。

## 6. 阶段一：1600训练、400验证

建议在 `tmux` 中启动：

```bash
tmux new -s aic-m-stage1
python -m scripts.train --config configs/train_fusion_v2_m_4090.yaml --yes
```

配置：

- YOLO26m-P2 + V2逐通道三模态融合
- `epochs=180`
- `patience=35`
- `imgsz=960`
- `batch=8`
- `workers=4`
- AMP
- 多尺度、Mosaic、MixUp、稀有类别权重

输出：

```text
outputs/aic_fusion_v2_m_stage1_4090/weights/best.pt
outputs/aic_fusion_v2_m_stage1_4090/weights/last.pt
outputs/aic_fusion_v2_m_stage1_4090/results.csv
```

如果进程意外中断且 `last.pt` 存在：

```bash
python -m scripts.train \
  --config configs/train_fusion_v2_m_4090.yaml \
  --resume outputs/aic_fusion_v2_m_stage1_4090/weights/last.pt \
  --yes
```

不要对正常完成并已经剥离优化器状态的 `last.pt` 使用恢复配置。

## 7. 1280显存冒烟与微调

阶段一完成后，先测试1280、`batch=4`：

```bash
python -m scripts.train --config configs/smoke_fusion_v2_m_highres_4090.yaml --yes
```

若发生 OOM，把以下三个配置的 `batch` 从4改为3：

```text
configs/smoke_fusion_v2_m_highres_4090.yaml
configs/finetune_fusion_v2_m_highres_4090.yaml
configs/final_all_fusion_v2_m_4090.yaml
```

冒烟正常后启动阶段二：

```bash
python -m scripts.train --config configs/finetune_fusion_v2_m_highres_4090.yaml --yes
```

输出：

```text
outputs/aic_fusion_v2_m_highres_4090/weights/best.pt
outputs/aic_fusion_v2_m_highres_4090/weights/last.pt
```

意外中断恢复：

```bash
python -m scripts.train \
  --config configs/finetune_fusion_v2_m_highres_4090.yaml \
  --resume outputs/aic_fusion_v2_m_highres_4090/weights/last.pt \
  --yes
```

阶段二仍使用固定400张验证集，依据 `best.pt` 判断1280是否提升。不要查看或人工调整测试集。

## 8. 阶段三：全部2000张训练图

只有阶段一和阶段二选择已经冻结后才能执行：

```bash
python -m scripts.train --config configs/final_all_fusion_v2_m_4090.yaml --yes
```

阶段三使用 train+val 共2000张训练图，训练20轮，不再根据这部分数据选择超参数。

意外中断恢复：

```bash
python -m scripts.train \
  --config configs/final_all_fusion_v2_m_4090.yaml \
  --resume outputs/aic_fusion_v2_m_final_all_4090/weights/last.pt \
  --yes
```

最终推理固定使用：

```text
outputs/aic_fusion_v2_m_final_all_4090/weights/last.pt
```

## 9. 自动TTA推理

```bash
python -m scripts.predict --config configs/predict_fusion_v2_m_tta_4090.yaml
```

它会对全部1000张测试图自动运行960、1280和1280水平翻转三个视图，并自动融合同类框。
不读取测试标签，不允许人工补框、删框或挑选单张结果。

输出目录：

```text
outputs/test_predictions_fusion_v2_m_tta_4090
```

## 10. 严格校验并打包

```bash
python -m scripts.package_submission \
  --predictions outputs/test_predictions_fusion_v2_m_tta_4090 \
  --test-images data/processed/images/test \
  --output outputs/submission_fusion_v2_m_tta_4090.zip
```

脚本会检查：

- 预测文件与1000张测试图一一对应；
- 没有缺失或额外 TXT；
- 每行恰好六列；
- 类别、归一化坐标和置信度合法；
- 每图不超过100个框；
- 每个文件按置信度降序排列；
- ZIP 根目录直接包含 TXT。

最终提交：

```text
outputs/submission_fusion_v2_m_tta_4090.zip
```

## 11. 必须备份

租赁实例释放前下载：

```text
outputs/aic_fusion_v2_m_stage1_4090/
outputs/aic_fusion_v2_m_highres_4090/
outputs/aic_fusion_v2_m_final_all_4090/
outputs/test_predictions_fusion_v2_m_tta_4090/
outputs/submission_fusion_v2_m_tta_4090.zip
```

并记录：

```bash
git rev-parse HEAD
python --version
python -m pip freeze
nvidia-smi
```
