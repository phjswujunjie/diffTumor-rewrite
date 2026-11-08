# 作者口径的分割训练（FM vs DDPM）

本目录新增：

- `train_seg_author.py`：参考 `STEP3/main.py` + `monai_trainer.py` 重写的训练入口
- `run_author.py`：依次启动 real / fm / ddpm 三组

## 与作者代码一致的部分

| 项目 | 实现 |
|---|---|
| 裁剪 | `RandCropByPosNegLabeld_select`，健康 CT 按 `fg_thresh` 重试 |
| Transform | 作者的完整 train/val transform 链 |
| 模型 | 3D U-Net `(16,32,64,128,256)`，`num_res_units=2` |
| Loss | `DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)` |
| 优化器 | AdamW，lr 2e-4，weight_decay 1e-5 |
| Scheduler | `LinearWarmupCosineAnnealingLR`，warmup 100 epochs |
| 在线合成 | 健康 CT 50% 概率注入肿瘤；80% early / 10% medium / 10% large |
| 验证 | sliding window 96³，overlap 0.75，organ pseudo + `denoise_pred` + Dice |
| Checkpoint | `model.pt`（best tumor Dice）/ `model_final.pt`（每 epoch） |

## 与作者代码的差异

1. **early 合成器可切换**
   - `--synt_model fm`：FM 10k，8 步
   - `--synt_model ddpm`：DDPM 10k，4 步
2. **medium/large 固定用作者的 `liver_noearly.pt`**
   - 两组完全相同，只比较 early 合成器
3. **合成器只加载一次**
   - 作者每个 epoch 重新加载；这里训练开始时加载，节省时间
4. **DataLoader 独立 seeded generator**
   - 保证 FM / DDPM 两组的数据顺序一致
5. **训练前重置 random / numpy RNG**
   - 保证两组生成的几何 mask、注入决策、羽化 sigma 配对
6. **val_epoch 额外计算**
   - tumor NSD(2mm)
   - tumor-wise sensitivity
7. **`--arm real` 过滤健康 CT 行**
   - real-only 只用 95 例真实肿瘤病例
8. **训练集缓存从内存改为磁盘（Windows 必需）**
   - `CacheDataset` 在 Windows 的 spawn 下会把几十 GB 缓存序列化给每个 worker，直接崩
   - 改用 `PersistentDataset`：只把「第一个随机变换之前」的确定性预处理写盘
   - 随机裁剪 / 翻转 / 旋转 / 强度扰动仍然每个 epoch 重新执行，语义与作者一致
   - 标签在缓存中存 `uint8`，载入后立即转回 `float32`，训练 dtype 不变
   - 三个 arm 共享同一缓存目录 `flowmatching/cache_persistent/`（transform 相同）

## 运行

```bash
cd STEP3.SegmentationModel

# 只打印命令，不训练
python flowmatching/run_author.py --arm all --dry-run

# smoke：1 epoch，5 iter
python flowmatching/run_author.py --arm real --smoke

# pilot：300 epoch
python flowmatching/run_author.py --arm all --max-epochs 300 --val-every 100

# 正式：2000 epoch
python flowmatching/run_author.py --arm all --max-epochs 2000 --val-every 200

# 断点续训（从各 arm 的 model_final.pt 继续）
python flowmatching/run_author.py --arm fm --max-epochs 300 --resume auto

# 作者的内存缓存（Windows 必须 workers=0，否则 spawn 序列化会崩）
python flowmatching/run_author.py --arm fm --max-epochs 300 \
    --cache-mode memory --cache-rate 0.5 --workers 0
```

## 输出

```
flowmatching/runs_author/{real,fm,ddpm}/
├── model.pt
├── model_final.pt
├── val_metrics.csv
└── events.out.tfevents...
```

## 注意

- **首次运行会把确定性预处理写盘**（约 0.46 GB/例，211 例约 97 GB），
  第一个 epoch 明显变慢，之后每个 epoch 直接读缓存。
- 缓存目录默认 `flowmatching/cache_persistent/`，三个 arm 共用；
  如果改了 `_get_transform` 的确定性部分，**必须删掉该目录**重建，否则会读到旧缓存。
- `--cache_mode` 可选 `disk`（默认）/ `memory` / `none`。
  `memory` 只在 Linux/fork 下建议使用；Windows 上 worker 会因序列化几十 GB 缓存而崩溃。
- **断点续训**：`--resume auto` 从 `<logdir>/model_final.pt` 继续。
  本次改动之后保存的 checkpoint 会额外包含 optimizer / scheduler（约 57 MB），
  可以精确恢复；更早的 19 MB checkpoint 只能恢复权重，AdamW 动量会重置。
- `--cache_mode memory` 是作者原版的内存缓存，但 Windows 上必须 `--workers 0`，
  否则 spawn 会把几十 GB 缓存序列化给 worker 而崩溃。