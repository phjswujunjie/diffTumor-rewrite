# Flow Matching 合成数据测试报告

日期：2026-11-04
环境：`D:\anaconda\envs\difftumor\python.exe`（Python 3.10.21, torch 2.7.1+cu128, RTX 5080 16GB）

## 1. 测试数据

人工合成，不依赖真实数据集：

- 3 个病例，`128 × 128 × 128`，1mm 各向同性，affine 为单位矩阵
- CT：高斯噪声 `N(40, 30)`，器官区域 +30 HU，clip 到 `[-175, 250]`
- 标签：`1 = 器官`（大椭球），`2 = 肿瘤`（半径 12 的球，位于器官内）
- 清单：`manifest/liver_tumor_data_early_fold/real_tumor_train_0.txt`

位置：`E:\thesisAndCode\data\_synth_fm_test`

## 2. 测试结果

### 2.1 数据管线 + 训练

| 实验 | 步数 | loss 起点 | loss 终点 |
|---|---|---|---|
| FM，连续 t | 20 | 0.9003 | **0.8112** |
| FM，离散 t | 2 | 0.8773 | 0.8678 |
| DDPM 基线 | 20 | 0.8729 | **0.7965** |

> 两者 loss 的绝对数值不可直接比较（FM 预测速度，DDPM 预测噪声），
> 这里只验证趋势和代码可运行性。

### 2.2 checkpoint

- 保存文件：`runs/synth/synth_fm/1.pt`
- 键：`['ema', 'model', 'scaler', 'step']`，与 DDPM 完全一致
- 大小：277MB（包含 VQGAN 权重，与 DDPM 的 checkpoint 一致）
- `ddpm.Tester.load()` 能同时恢复 `model` 和 `ema`

### 2.3 采样

| 检查 | 结果 |
|---|---|
| `sample()` 输出 | `(1, 1, 96, 96, 96)` |
| 数值有限性 | 无 NaN / Inf |
| `sample_latent(steps=2/4/8)` | U-Net 调用次数 = 2 / 4 / 8 |
| 连续 t 默认 4 步 | 4 NFE |
| 离散 t（T=4） | 3 NFE（T-1） |
| `Tester.ema_model.sample()` | 正常 |

## 3. 复现命令

```bash
cd E:/thesisAndCode/code/DiffTumor/STEP2.DiffusionModel

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
D:/anaconda/envs/difftumor/python.exe -W ignore flowmatching/train.py \
  dataset.name=synth dataset.fold=0 \
  dataset.dataset_list='[liver_tumor_data_early_fold]' \
  dataset.data_root_path='E:/thesisAndCode/data/_synth_fm_test/' \
  dataset.label_root_path='E:/thesisAndCode/data/_synth_fm_test/' \
  dataset.data_txt_path='E:/thesisAndCode/data/_synth_fm_test/manifest/' \
  dataset.cache_dataset=False dataset.uniform_sample=False dataset.num_samples=1 \
  dataset.batch_size=1 dataset.num_workers=0 \
  model.vqgan_ckpt='E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt' \
  model.batch_size=1 model.train_num_steps=20 model.gradient_accumulate_every=1 \
  model.save_and_sample_every=1000 model.num_workers=0 model.gpus=0 model.amp=False \
  model.results_folder='E:/thesisAndCode/data/_synth_fm_test/runs/' \
  model.results_folder_postfix=synth_fm_20
```

DDPM 基线把入口换成 `train.py`，`model.results_folder` 换成 `runs_ddpm/` 即可。

## 4. 结论

- 数据加载 → VQGAN 编码 → 条件构造 → FM 训练 → checkpoint → EMA 采样，全链路通过
- 连续/离散两种时间参数化均正常
- 采样 NFE 与配置一致
- 与 DDPM 共用同一套 Unet/Trainer/Tester/数据管线，未出现形状或设备错误