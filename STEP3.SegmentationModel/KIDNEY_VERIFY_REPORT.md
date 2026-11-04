# 肾脏早期肿瘤合成 —— 端到端验证报告

日期：2026-10-23
脚本：`verify_kidney_synthesis.py`
数据：`E:\thesisAndCode\data\HealthyCT\extracted\healthy_ct`
权重：`TumorGeneration/model_weight/{AutoencoderModel.ckpt, kidney_early.pt, kidney_noearly.pt}`

## 1. 结论

**官方 kidney 管线在真实肾脏 CT 上可以正常跑通。**

- 24 次合成（12 病例 × 2 种子）全部通过
- 标签恒为 `{0,1,2}`，肿瘤体素数 > 0
- 肿瘤全部落在器官内；`label1 + label2` 恒等于输入器官体素数
- 羽化掩膜支持域之外，原图逐体素完全未变（`changed_outside_support_voxels = 0`）
- 保存的 5 个 NIfTI 共享同一个 crop affine，空间一致

> 注意：`kidney_ct` 里**没有 CHAOS**。CHAOS 只出现在 `liver_label`。
> 肾脏数据源为 `01_Multi-Atlas`(11)、`07_WORD`(16)、`08_AbdomenCT-1K`(93)。

## 2. 环境

| 项 | 值 |
|---|---|
| conda 环境 | `D:\anaconda\envs\difftumor\python.exe`（Git Bash 里 conda 不在 PATH） |
| Python | 3.10.21 |
| PyTorch | 2.7.1+cu128 |
| GPU | RTX 5080 16 GB（sm_120） |
| MONAI / Hydra | 0.9.0 / 1.2.0 |
| kidney_early.pt | step = 59248 |
| kidney_noearly.pt | step = 57869 |

## 3. 实现方式

预处理严格对齐 `main.py` 的 `_get_transform`：

```
LoadImaged -> AddChanneld -> Orientationd(RAS) -> Spacingd(1mm)
-> ScaleIntensityRanged(HU[-175,250] -> [0,1])
-> 肾脏标签 >0 合并为 1（官方对 kidney_label 的处理）
-> 96^3 裁剪
```

裁剪默认复刻 `RandCropByPosNegLabeld_select` 的行为：
随机前景中心 + `fg_thresh=25mm`，前景不足时按 5 次一档放宽到 20/15/10/5mm。
（`--crop-mode max` 可切换为"500 候选取前景最多"的备选策略。）

合成直接调用未修改的官方函数：

```python
vqgan, early_tester, _ = synt_model_prepare(device, organ="kidney")
gen_ct, gen_lbl = synthesize_early_tumor(ct, organ_mask, "kidney", vqgan, early_tester)
```

随机种子同时设置 `random` / `numpy` / `torch`，保证可复现。

## 4. 统计结果（24 次）

| 指标 | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| 肿瘤体素数 | 113 | 375 | 509 | 859 | 4152 |
| 肿瘤/周围肾脏 HU 差 | 10.1 | 29.0 | 42.2 | 72.6 | 134.5 |
| 连通病灶数 | 1 | 1 | 2 | 3 | 4 |

- 肿瘤体素中位数 ≈ 509，等效球直径约 10mm，符合 tiny/small 的设定
- 24/24 未触碰裁剪边界
- 5/24 的 HU 对比 < 20（视觉上很淡），7/24 < 30
- 2/24 的肿瘤平均 HU **高于**周围肾脏，其余为低密度

连通域最多到 4，是因为 tiny/small 允许 1~3 个病灶，且形变/器官表面交集可能把一个病灶切成多块，不是 bug。

## 5. 质量提示

官方管线**不做任何对比度过滤**。约 20% 的合成肿瘤与周围组织对比 < 20 HU，
若直接用于训练分割网络，这些样本接近"不可见目标"，可能稀释监督信号。

建议（尚未实现）：
- 在合成后按 `|tumor_mean_HU - ring_mean_HU|` 过滤，例如阈值 20~25 HU
- 或记录该指标，训练时按对比度加权

## 6. 复现

```bash
cd STEP3.SegmentationModel
PY=D:/anaconda/envs/difftumor/python.exe

# 单病例
$PY -W ignore verify_kidney_synthesis.py

# 跨数据源批量（12 病例 × 2 种子）
$PY -W ignore verify_kidney_synthesis.py --batch 12 --seeds 2

# 只看指标、不写文件
$PY -W ignore verify_kidney_synthesis.py --batch 12 --seeds 2 --no-save
```

产物目录：`E:\thesisAndCode\data\HealthyCT\generated_kidney_verify\<case>/`

- `kidney_ct_crop_norm.nii.gz` —— [0,1] 归一化 CT
- `kidney_organ_mask_crop.nii.gz` —— 器官掩膜 {0,1}
- `synthetic_ct_early_norm.nii.gz` —— 合成 CT [0,1]
- `synthetic_ct_early_clippedHU.nii.gz` —— 合成 CT，还原到 [-175,250]
- `synthetic_kidney_tumor_label.nii.gz` —— {0=背景, 1=肾脏, 2=肿瘤}
- `preview.png` / `diagnostics.json`

## 7. 对下一步（B：STEP3 训练）的影响

1. 合成链路已验证，可以放心让 `monai_trainer.py` 在线调用。
2. 16 GB 显存跑官方命令（`--noamp`, `batch_size=12`）不现实，需要降 batch / 开 AMP。
3. `main.py` 对 kidney 的真实肿瘤病例做 `label==3 -> 1` 的处理，
   接入 MSD-Kidney 前应先确认其标签约定（1=肾, 2=肿瘤, 3=?），否则可能把肿瘤误并成器官。