# AIC 2026 城市场景多模态目标检测

当前推荐的高分路线是 YOLO26m-P2 + V2 逐通道多模态融合。单卡 RTX 4090
从环境检查、冒烟、三阶段训练到TTA打包的唯一当前入口见
`YOLO26M_V2_4090_RUNBOOK.md`。旧的 YOLO26s/V1 配置保留用于复现实验，不应与
新路线混用。

本项目实现单模型、全自动的 RGB/红外/深度目标检测流水线。它严格区分官方原始数据和派生数据：原始目录只读，训练/验证划分、越界框修正、8 通道编码、推理结果均可追溯生成；测试集不参与标注、调参或人工编辑。

## 方案结构

- `aic_mm/data`：数据审计、场景分组划分、8 通道编码、模态增强
- `aic_mm/models`：RGB/红外/深度三分支质量门控融合 stem
- `aic_mm/training`：YOLO26m-P2/V2 自定义训练器
- `aic_mm/evaluation`：本地训练验证集 101 点插值 mAP
- `aic_mm/inference`：测试集自动推理、格式校验和 ZIP 打包
- `configs`：预处理、第一阶段训练、高分辨率微调和推理配置
- `scripts`：各阶段独立命令行入口
- `tests`：标签、编码、指标和提交格式单元测试

固定的 8 通道顺序为：RGB 三通道、原始红外、对比度红外、近距离深度、深度有效性、深度制式标记。最后一个通道明确区分 16 位公制 PNG 深度和 8 位 JPG 伪深度，避免网络混淆两类数值语义。

## 依赖与权重

当前 `aic-mm` 环境中的 PyTorch 2.11.0+cu128、Ultralytics 8.4.106 已满足主要要求。运行命令前先确认终端提示符以 `(aic-mm)` 开头，并在项目根目录执行：

```bash
mkdir -p weights
python -c "from ultralytics.utils.downloads import attempt_download_asset; print(attempt_download_asset('weights/yolo26m.pt'))"
```

这会通过已安装 Ultralytics 的官方资源下载器获取公开预训练权重。比赛任务书允许使用公开预训练权重；训练数据仍只使用官方训练集。

开发测试需要 `pytest`。如果当前环境没有：

```bash
python -m pip install pytest
```

高质量划分使用近重复图像哈希检查，需要：

```bash
python -m pip install ImageHash scipy
```

当前环境已经检测到 `ImageHash 4.3.2` 和 `SciPy 1.15.3`。后者用于整数规划，使训练/验证集同时满足场景组隔离、类别比例和图像域比例；若换环境，二者都应保留。

## 推荐执行顺序

先做轻量、只读的数据审计和分组划分：

```bash
python -m scripts.audit_data --compute-phash
python -m scripts.make_split
```

然后生成派生的 8 通道 TIFF。该步骤处理 3000 张全分辨率多模态样本，会耗时并占用较多磁盘；`--yes` 是显式确认：

```bash
python -m scripts.build_multispectral --workers 2 --yes
```

RTX 4090 上先执行预检和冒烟：

```bash
python -m scripts.preflight_4090
python -m scripts.train --config configs/smoke_fusion_v2_m_4090.yaml --yes
```

冒烟通过后依次运行三阶段训练：

```bash
python -m scripts.train --config configs/train_fusion_v2_m_4090.yaml --yes
python -m scripts.train --config configs/smoke_fusion_v2_m_highres_4090.yaml --yes
python -m scripts.train --config configs/finetune_fusion_v2_m_highres_4090.yaml --yes
python -m scripts.train --config configs/final_all_fusion_v2_m_4090.yaml --yes
```

测试推理和提交打包：

```bash
python -m scripts.predict --config configs/predict_fusion_v2_m_tta_4090.yaml
python -m scripts.package_submission \
  --predictions outputs/test_predictions_fusion_v2_m_tta_4090 \
  --test-images data/processed/images/test \
  --output outputs/submission_fusion_v2_m_tta_4090.zip
```

`package_submission.py` 会检查恰好 1000 个 TXT、每行六列、类别/坐标/置信度范围、每图最大目标数和置信度排序，并保证 ZIP 根目录直接放置结果文件。不要手动修改任何测试结果。

## RTX 4090 长时间训练与恢复

当前路线不使用旧的 `supervisor_4090d_clean.sh`。在 `tmux` 中逐阶段执行，并在意外
中断时对同一个正式配置添加：

```bash
--resume outputs/对应阶段/weights/last.pt
```

完整命令、输出目录、显存降级顺序和备份清单见
`YOLO26M_V2_4090_RUNBOOK.md`。

## 验证与调参纪律

所有阈值、分辨率和增强参数只根据固定训练验证集决定。需要验证某个权重时，可让 Ultralytics 在 `data/processed/aic_multispectral.yaml` 的 `val` 集上输出 TXT，再运行：

```bash
python -m scripts.evaluate_predictions --predictions path/to/validation_txt
```

禁止根据测试图像内容人工补框、删框、改框或选择单张结果；也不要把测试结果反馈进训练流程。最终只能提交代码自动生成的单模型结果。

## 轻量测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```
