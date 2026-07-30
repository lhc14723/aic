# AIC 多模态检测 V2 训练策略

## 结论

不建议立即删除现有模型，也不建议只在现有模型上无限续训。

项目现在有两条隔离的实验路线：

1. **V1 现有权重增强**：从当前官网 46.644 分对应权重继续做 1280
   高分辨率、低学习率全数据微调。这条路线成本低，适合先产生一个可提交候选。
2. **V2 干净重训**：从公开的 `yolo26m.pt` 预训练权重开始，在固定
   train/val 划分上训练新的逐通道多模态门控。这条路线成本高，但能够可靠选模，
   也是冲击更高上限的主路线。

这里的“重训”不是随机初始化。YOLO 主干继续使用公开预训练参数，并且训练器会
显式把官方 RGB 首层卷积复制到 RGB 分支，再用其均值初始化红外、深度分支。

## 为什么这样修改

960 验证集消融结果为：

| 输入模态 | mAP50-95 | 相对完整模态 |
|---|---:|---:|
| RGB + 红外 + 深度 | 0.4843 | 0 |
| 仅 RGB | 0.4561 | -0.0282 |
| RGB + 红外 | 0.4776 | -0.0068 |
| RGB + 深度 | 0.4671 | -0.0172 |

红外贡献约 1.72 个百分点，深度贡献约 0.68 个百分点。因此 V2 不再让辅助模态
所有特征通道共用一个门值，而是对每个特征通道分别预测红外和深度权重。配置也
降低了模态丢弃和深度挖洞强度，避免把始终存在的官方传感器信息过度抹除。

类别权重指数从 0.15 提高到 0.25，重点改善 `boat`、`ball`、
`garbagecan`、`tricycle` 等少样本类别；同时提高 box/DFL 权重并加入温和的
多尺度训练，针对比赛的 mAP50-95 定位精度。

## 路线 A：先提升现有模型

当前本地 8 GB 显卡已经实测可运行 `imgsz=1280, batch=2, AMP`。

正式训练：

```bash
python -m scripts.train \
  --config configs/upgrade_existing_highres_all.yaml \
  --yes
```

训练完成后使用同一模型的多尺度/水平翻转 TTA 自动推理：

```bash
python -m scripts.predict \
  --config configs/predict_existing_upgraded_tta.yaml
```

如果暂时不续训，也可直接测试当前权重的 TTA：

```bash
python -m scripts.predict --config configs/predict_existing_tta.yaml
```

TTA 仍然是单模型自动推理，不读取测试标签，也不人工改框。项目显式运行
960、1280、1280 水平翻转三个视图，再按类别自动融合；这是因为 YOLO26 的
end-to-end 检测头不支持 Ultralytics 内置的 `augment=True`。代价是推理时间
大约增加到普通推理的三倍。生成后仍需运行原有严格审计和打包脚本。

这条路线使用了全部 2000 张训练图，并且源权重已经见过本地 val，因此其本地
val 指标不能用于判断泛化能力，只能以官网分数验证增益。

## 路线 B：V2 干净重训

先做一次只检查代码和显存的冒烟测试：

```bash
python -m scripts.preflight_4090
python -m scripts.train --config configs/smoke_fusion_v2_m_4090.yaml --yes
```

该配置只使用 2% 数据训练 1 轮，生成的权重禁止提交。

随后依次执行：

```bash
python -m scripts.train --config configs/train_fusion_v2_m_4090.yaml --yes
python -m scripts.train --config configs/smoke_fusion_v2_m_highres_4090.yaml --yes
python -m scripts.train --config configs/finetune_fusion_v2_m_highres_4090.yaml --yes
```

第一阶段的 `best.pt` 由固定、场景隔离的 400 张 val 选择；第二阶段继续用同一个
干净 val 做高分辨率定位微调。只有两阶段参数确定后，才执行全 2000 张训练：

```bash
python -m scripts.train --config configs/final_all_fusion_v2_m_4090.yaml --yes
python -m scripts.predict --config configs/predict_fusion_v2_m_tta_4090.yaml
```

## 4090 服务器调整

仓库中的配置首先采用本地安全参数。迁移到 4090/4090D 后，只调整资源参数，
不要同时改变学习率、增强和模型结构：

- `workers`: 使用 4；
- `batch`: 960 阶段从 6 开始，1280 阶段从 4 开始，并以冒烟结果决定是否下调；
- 保持 `amp: true`；
- 若显存不足，依次降低 batch，不要先降低 `imgsz`。

batch 改大后当前 AdamW 学习率不必线性放大；数据只有 2000 张，保持配置中的
保守学习率更稳妥。

## 结果判断

- V1 续训是低风险候选，但不能指望仅靠 1280 稳定增加 13 个百分点。
- V2 必须以干净 val 的 `mAP50-95` 与旧模型同阶段结果比较。
- 只保留清晰优于基线的单项改动；若 V2 没有提高，不进入全数据阶段。
- 当前主路线已经使用更大的 YOLO26m；若完整三阶段结果仍达不到目标，再根据固定
  验证集决定是否尝试蒸馏，不要根据测试集结果反向人工修改预测。

完整服务器环境、恢复命令、TTA和打包步骤以
`YOLO26M_V2_4090_RUNBOOK.md` 为准。
